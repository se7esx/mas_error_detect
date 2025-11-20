import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
d = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# ==========================
# 1. Sentence Encoder (BERT)
# ==========================
class SentenceEncoder(nn.Module):
    def __init__(self, model_name='/home/shenxu/all-MiniLM-L6-v2/', device="cuda"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name).to(device)
        self.device = device
        self.hidden_size = self.encoder.config.hidden_size

    @torch.no_grad()
    def encode(self, sentences):
        tokens = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(self.device)
        outputs = self.encoder(**tokens)
        embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings


# ======================================
# 2. Qwen3-8B 作为 Decoder 做重构
#    - 冻结 Qwen，仅训练 in/out 投影（默认）
#    - 用 inputs_embeds 喂入投影后的 step embeddings
# ======================================


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class QwenDecoderPredictor(nn.Module):
    def __init__(
        self,
        emb_dim,
        base_model="Qwen/Qwen3-8B",
        freeze_qwen=True,
        use_device_map_auto=True,
        torch_dtype=torch.bfloat16,
        num_heads=4
    ):
        super().__init__()
        load_kwargs = dict(trust_remote_code=True)
        if use_device_map_auto:
            load_kwargs.update(dict(device_map="auto", torch_dtype=torch_dtype))
        self.qwen = AutoModel.from_pretrained(base_model, **load_kwargs)

        if freeze_qwen:
            self.qwen.eval()
            for p in self.qwen.parameters():
                p.requires_grad = False

        self.qwen_hidden = self.qwen.config.hidden_size
        self.input_proj  = nn.Linear(emb_dim, self.qwen_hidden)
        self.output_proj = nn.Linear(self.qwen_hidden, emb_dim)
        self.out_dtype = torch.float32

        # === Prototype: 高斯初始化 (1, emb_dim)，训练时更新 ===
        proto_init = torch.randn(1, emb_dim) * 0.02
        self.prototype = nn.Parameter(proto_init)

        # === Attention 聚合模块（训练时用） ===
        self.attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, batch_first=True)

    def forward(self, embeddings, train_mode=True):
        """
        embeddings: (B, L, D)
        train_mode=True: 返回 (pred, proto_out)，proto_out用于loss更新prototype
        train_mode=False: 返回 (pred, self.prototype)，推理时只用learned prototype
        """
        device = embeddings.device
        self.input_proj = self.input_proj.to(device)
        self.output_proj = self.output_proj.to(device)
        self.attn = self.attn.to(device)
        #self.prototype = self.prototype.to(device)
        x = self.input_proj(embeddings)  # (B,L,Hq)
        x = x.to(dtype=getattr(self.qwen, "dtype", x.dtype))
        B, L, _ = x.size()
        attn_mask = torch.ones((B, L), dtype=torch.long, device=x.device)

        outputs = self.qwen(inputs_embeds=x, attention_mask=attn_mask)
        hidden  = outputs.last_hidden_state
        pred = self.output_proj(hidden.to(self.out_dtype))   # (B,L,D)
        proto = self.prototype.to(device)
        if train_mode:
            # === prototype 聚合 ===
            proto_expanded = proto.unsqueeze(0).expand(B, -1, -1)  # (B,1,D)
            proto_out, _ = self.attn(query=proto_expanded, key=pred, value=pred)  # (B,1,D)
            proto_out = proto_out.squeeze(1)  # (B,D)
            return pred, proto_out
        else:
            return pred, self.prototype.expand(B, -1) # (B,D)

def training_loss(pred, target, proto_out, target_seq, lambda_proto=0.1):
    """
    pred: (B,L,D)       —— 重建结果
    target: (B,L,D)     —— 真实序列
    proto_out: (B,D)    —— 注意力聚合的prototype
    target_seq: (B,L,D) —— 输入真实序列
    """
    # 重建损失
    recon_loss = F.mse_loss(pred, target)

    # 原型对齐损失: 与输入序列的均值对齐
    seq_mean = target_seq.mean(dim=1)  # (B,D)
    proto_loss = F.mse_loss(proto_out, seq_mean)

    return recon_loss + lambda_proto * proto_loss


# ==========================
# 3. 训练/推理
# ==========================


def train_epoch_with_proto(model, data, optimizer, device="cuda", lambda_proto=0.1):
    """
    在原始重建loss基础上，增加prototype对齐loss。
    """
    model.train()
    total_loss, count = 0.0, 0

    for seq in data:
        if seq.size(0) < 2:
            continue

        seq = seq.to(device).unsqueeze(0)  # (1,L,D)
        target = seq[:, 1:, :]              # (1,L-1,D)

        # forward: 返回 (pred, proto_out)
        pred, proto_out = model(seq[:, :-1, :], train_mode=True)  # pred:(1,L-1,D), proto_out:(1,D)

        if pred.numel() == 0:
            continue

        # (1) 重建loss
        recon_loss = F.mse_loss(pred, target)

        # (2) prototype对齐loss (对齐序列均值)
        seq_mean = target.mean(dim=1)   # (1,D)
        proto_loss = F.mse_loss(proto_out, seq_mean)

        loss = recon_loss + lambda_proto * proto_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1

    return total_loss / max(1, count)

import torch

def normalize_single_score_sigmoid(score, threshold=0.7):
    s = torch.tensor(score, dtype=torch.float32)
    normed = torch.sigmoid(s)   # 映射到 (0,1)
    label = int(normed > threshold)
    return normed.item(), label

import torch

def normalize_single_score(score, gamma=1.0, bias=0.0, threshold=0.5):
    s = torch.tensor(score, dtype=torch.float32)
    normed = torch.sigmoid(gamma * s + bias)
    label = int(normed > threshold)
    return normed.item(), label

# 示例：希望 score=2 对应 sigmoid 输出=0.5



@torch.no_grad()
def anomaly_score_autoregressive_with_proto(model, seq, device="cuda", alpha=1.0, beta=0.2):
    """
    自回归 + prototype 一致性
    """
    model.eval()
    seq = seq.to(device)    # (L,D)
    L = seq.size(0)
    scores = []
    if L < 2:
        return [1.0]

    history_pred = seq[0:1, :].unsqueeze(0)  # (1,1,D)
    proto = model.prototype.to(seq.device)
    proto = F.normalize(proto, dim=-1)  # (1,D)
    preds = []
    for t in range(0, L - 1):
        pred_all, _ = model(history_pred, train_mode=False)  # pred_all:(1,len(history),D)
        pred_next = pred_all[:, -1, :]  # (1,D)

        true_next = seq[t + 1, :].unsqueeze(0)  # (1,D)

        # (1) MSE
        mse_score = F.mse_loss(pred_next, true_next, reduction="mean").item()

        # (2) Prototype 偏离
        pred_norm = F.normalize(pred_next, dim=-1)  # (1,D)
        sim = torch.matmul(pred_norm, proto.T).squeeze().item()
        proto_score = 1 - sim

        score = alpha * mse_score + beta * proto_score
        #score, label =  normalize_single_score_sigmoid(score)

        scores.append(score)
        #preds.append(label)
        # 自回归扩展
        history_pred = torch.cat([history_pred, pred_next.unsqueeze(1)], dim=1)

    return scores


@torch.no_grad()
def anomaly_score_autoregressive_with_proto_pred(model, seq, device="cuda", alpha=1.0, beta=0.2):
    """
    自回归 + prototype 一致性
    """
    model.eval()
    seq = seq.to(device)  # (L,D)
    L = seq.size(0)
    scores = []
    if L < 2:
        return [1.0]

    history_pred = seq[0:1, :].unsqueeze(0)  # (1,1,D)
    proto = model.prototype.to(seq.device)
    proto = F.normalize(proto, dim=-1)  # (1,D)
    preds = []
    for t in range(0, L - 1):
        pred_all, _ = model(history_pred, train_mode=False)  # pred_all:(1,len(history),D)
        pred_next = pred_all[:, -1, :]  # (1,D)

        true_next = seq[t + 1, :].unsqueeze(0)  # (1,D)

        # (1) MSE
        mse_score = F.mse_loss(pred_next, true_next, reduction="mean").item()

        # (2) Prototype 偏离
        pred_norm = F.normalize(pred_next, dim=-1)  # (1,D)
        sim = torch.matmul(pred_norm, proto.T).squeeze().item()
        proto_score = 1 - sim

        #score = alpha * mse_score + beta * proto_score
        score = proto_score
        score, label = normalize_single_score_sigmoid(score, 0.5)
        #gamma = 1.0
        #bias = -1.0 * gamma
        #score, label = normalize_single_score(score,gamma, bias)

        scores.append(score)
        preds.append(label)
        # 自回归扩展
        history_pred = torch.cat([history_pred, pred_next.unsqueeze(1)], dim=1)

    return scores,preds


def anomaly_two_score_autoregressive_with_proto_pred(model, seq, device="cuda", alpha=0.9, beta=0.1):
    """
    自回归 + prototype 一致性
    """
    model.eval()
    seq = seq.to(device)  # (L,D)
    L = seq.size(0)
    scores = []
    if L < 2:
        return [1.0]

    history_pred = seq[0:1, :].unsqueeze(0)  # (1,1,D)
    proto = model.prototype.to(seq.device)
    proto = F.normalize(proto, dim=-1)  # (1,D)
    preds = []
    s1 = []
    p1 = []
    s_all = []
    p_all =[]
    for t in range(0, L - 1):
        pred_all, _ = model(history_pred, train_mode=False)  # pred_all:(1,len(history),D)
        pred_next = pred_all[:, -1, :]  # (1,D)

        true_next = seq[t + 1, :].unsqueeze(0)  # (1,D)

        # (1) MSE
        mse_score = F.mse_loss(pred_next, true_next, reduction="mean").item()
        score_1, label_1 = normalize_single_score_sigmoid(mse_score, 0.5)
        s1.append(score_1)
        p1.append(label_1)
        # (2) Prototype 偏离
        pred_norm = F.normalize(pred_next, dim=-1)  # (1,D)
        sim = torch.matmul(pred_norm, proto.T).squeeze().item()
        proto_score = 1 - sim


        #score = proto_score
        score, label = normalize_single_score_sigmoid(proto_score, 0.5)
        #gamma = 1.0
        #bias = -1.0 * gamma
        #score, label = normalize_single_score(score,gamma, bias)

        scores.append(score)
        preds.append(label)
        score = alpha * mse_score + beta * proto_score
        score_all, label_all = normalize_single_score_sigmoid(score, 0.5)
        s_all.append(score_all)
        p_all.append(label_all)
        # 自回归扩展
        history_pred = torch.cat([history_pred, pred_next.unsqueeze(1)], dim=1)

    return s_all,p_all,s1,p1,scores,preds
def train_epoch(model, data, optimizer, device="cuda"):
    model.train()
    total_loss = 0.0
    count = 0
    for seq in data:
        if seq.size(0) < 2:   # 只有一个 step → 跳过
            continue

        seq = seq.to(device).unsqueeze(0)
        pred   = model(seq[:, :-1, :])    # (1,L-1,D)
        target = seq[:, 1:, :]            # (1,L-1,D)

        if pred.numel() == 0:  # 空预测，跳过
            continue

        loss = F.mse_loss(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        count += 1

    return total_loss / max(1, count)



@torch.no_grad()
def anomaly_score_autoregressive(model, seq, device="cuda"):
    """
    自回归 next-step reconstruction：
      - 初始历史 = seq[0]
      - 逐步预测下一个 \hat{h}_{t+1}，与真实 h_{t+1} 做 MSE，作为该步的 anomaly score
      - 历史扩展时使用 "预测得到的向量"（更贴近部署）
    返回：长度为 L-1 的分数列表（对应 step 1..L-1）
    """
    model.eval()
    seq = seq.to(device)                     # (L, D)
    L   = seq.size(0)
    scores = []
    if L < 2:   # 只有一个 step，没有 anomaly score
        return [1.0]

    # history_pred: (1, 1, D)，仅含首步（用真值启动）
    history_pred = seq[0:1, :].unsqueeze(0)

    for t in range(0, L - 1):
        # 预测基于当前历史的“下一步”表征
        pred_all = model(history_pred)       # (1, len(history), D)
        pred_next = pred_all[:, -1, :]       # 取最后一个位置的预测，(1, D)

        true_next = seq[t + 1, :].unsqueeze(0)  # (1, D)
        score = F.mse_loss(pred_next, true_next, reduction="mean").item()
        scores.append(score)

        # 自回归历史扩展：把 "预测的向量" 拼到历史后面
        history_pred = torch.cat([history_pred, pred_next.unsqueeze(1)], dim=1)

    return scores

@torch.no_grad()
def anomaly_score_teacher_forcing_with_proto(model, seq, device="cuda", alpha=1.0, beta=0.2, threshold=0.5):
    """
    Teacher-forcing + prototype 一致性：
      - 每一步预测基于真实历史
      - 异常分数 = α * MSE + β * Prototype 偏离
      - 返回：raw scores, sigmoid-normalized scores, 0/1 preds
    """
    model.eval()
    seq = seq.to(device)   # (L, D)
    L   = seq.size(0)

    scores = []
    sigmoid_score = []
    preds = []

    # 🔑 prototype 对齐到输入的 device
    proto = model.prototype.to(seq.device)
    proto = F.normalize(proto, dim=-1)  # (1,D)

    for t in range(0, L - 1):
        # (1) 基于真实历史预测下一步
        history_true = seq[:t+1, :].unsqueeze(0)   # (1, t+1, D)
        pred_all, _ = model(history_true, train_mode=False)   # pred_all:(1,t+1,D)
        pred_next = pred_all[:, -1, :]             # (1, D)

        true_next = seq[t + 1, :].unsqueeze(0)     # (1, D)

        # (2) MSE 部分
        mse_score = F.mse_loss(pred_next, true_next, reduction="mean").item()

        # (3) Prototype 偏离 (预测 vs prototype)
        pred_norm = F.normalize(pred_next, dim=-1)   # (1,D)
        sim = torch.matmul(pred_norm, proto.T).squeeze().item()
        proto_score = 1 - sim

        # (4) 综合分数
        score = alpha * mse_score + beta * proto_score
        scores.append(score)

        # (5) 映射到 [0,1]，并用阈值生成 0/1 标签
        s_score = torch.sigmoid(torch.tensor(score)).item()
        pred = int(s_score > threshold)

        sigmoid_score.append(s_score)
        preds.append(pred)

    return scores, sigmoid_score, preds


@torch.no_grad()
def anomaly_score_teacher_forcing(model, seq, device="cuda"):
    """
    Teacher-forcing 推理：
      - 每一步预测都是基于真实历史 step
      - 不再使用自回归拼接预测 embedding
      - 返回：长度为 L-1 的分数列表
    """
    model.eval()
    seq = seq.to(device)   # (L, D)
    L   = seq.size(0)
    scores = []
    sigmoid_score = []
    preds = []
    for t in range(0, L - 1):
        # 基于真实历史 [0..t] 预测下一步
        history_true = seq[:t+1, :].unsqueeze(0)   # (1, t+1, D)
        pred_all = model(history_true)             # (1, t+1, D)
        pred_next = pred_all[:, -1, :]             # (1, D)

        true_next = seq[t + 1, :].unsqueeze(0)     # (1, D)
        score = F.mse_loss(pred_next, true_next, reduction="mean").item()
        scores.append(score)
        s_score,pred = normalize_single_score_sigmoid(score, 0.5)
        preds.append(pred)
        sigmoid_score.append(s_score)
    return scores,sigmoid_score

import os
import json


def build_dataset_from_single_files_windows(folder_path, file_list, window_size=3):
    all_sentences = []
    file_used = []

    for filename in file_list:
        file_path = os.path.join(folder_path, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

            mistake_step = int(data.get("mistake_step", -1))
            if mistake_step == 0:
                continue

            # 为每个文件创建多个数据点
            file_data_points = []

            if "Hand-Crafted" not in folder_path:
                query = data.get("question", "")

            history = data.get("history", [])

            if mistake_step > 0 and len(history) > 0:
                end_index = min(mistake_step + 1, len(history))

                # 滑动窗口提取
                for start_idx in range(0, end_index - window_size + 1, window_size):
                    window_end = start_idx + window_size
                    if window_end > end_index:
                        break

                    # 构建一个数据点
                    data_point = []
                    if "Hand-Crafted" not in folder_path:
                        data_point.append(query)

                    # 添加窗口内容
                    for i in range(start_idx, window_end, window_size):
                        if i < len(history):
                            data_point.append(history[i]["content"])

                    file_data_points.append(data_point)

            # 添加到总数据集
            for data_point in file_data_points:
                all_sentences.append(data_point)
                file_used.append(filename)

    return all_sentences

def build_dataset_from_single_files(folder_path, file_list,test_is = False):
    all_sentences = []
    all_labels = []
    file_used = []

    for filename in file_list:
        file_path = os.path.join(folder_path, filename)
        st = []
        s_label = []
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "Hand-Crafted" not in folder_path:
                query = data.get("question", [])
                st.append(query)
                s_label.append(0)
            history = data.get("history", [])
            mistake_step = int(data.get("mistake_step", -1))
            if not test_is:
                # 正确构建label
                for i in range(mistake_step):  # 不包括错误步
                    st.append(history[i]["content"])
                    s_label.append(1 if i == mistake_step else 0)
                s_label.append(1)
            else:
                for i in range(mistake_step+1):  # 包括错误步
                    st.append(history[i]["content"])
                    s_label.append(1 if i == mistake_step else 0)
            file_used.append(mistake_step)
        all_sentences.append(st)
        all_labels.append(s_label)
    return all_sentences, all_labels, file_used


def build_dataset_from_single_all_sentence_files(folder_path, file_list,test_is = False):
    all_sentences = []
    all_labels = []
    file_used = []

    for filename in file_list:
        file_path = os.path.join(folder_path, filename)
        st = []
        s_label = []
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "Hand-Crafted" not in folder_path:
                query = data.get("question", [])
                st.append(query)
                s_label.append(0)
            history = data.get("history", [])
            mistake_step = int(data.get("mistake_step", -1))
            if not test_is:
                # 正确构建label
                for i in range(mistake_step):  # 不包括错误步
                    st.append(history[i]["content"])
                    s_label.append(1 if i == mistake_step else 0)
                s_label.append(1)
            else:
                for i in range(len(history)):  # 包括错误步
                    st.append(history[i]["content"])
                    s_label.append(1 if i == mistake_step else 0)
            file_used.append(mistake_step)
        all_sentences.append(st)
        all_labels.append(s_label)
    return all_sentences, all_labels, file_used
# ==========================
# 4. Demo
# ==========================
import numpy as np


def minmax_normalize(scores):
    """Min-Max归一化到0-1之间"""
    scores_array = np.array(scores)
    min_val = np.min(scores_array)
    max_val = np.max(scores_array)

    # 避免除零错误
    if max_val == min_val:
        return [0.5] * len(scores)  # 所有值相同时返回0.5

    normalized_scores = (scores_array - min_val) / (max_val - min_val)
    return normalized_scores.tolist()
if __name__ == "__main__":
    # 你也可以把 device 固定成 "cuda" 并手动关闭 device_map
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) 编码器（如需换成同 family，可替换为 Qwen-Embedding）
    encoder = SentenceEncoder(device=device)
    #folder_path = "/home/shenxu/agent_failure/Who&When/Algorithm-Generated/"
    folder_path = "/home/shenxu/agent_failure/Who&When/Algorithm-Generated/"
    # train_files = [f for f in os.listdir(folder_path)
    #               if os.path.isfile(os.path.join(folder_path, f))]
    SILDE_WINDOW = False
    if SILDE_WINDOW:
        train_files = [f for f in os.listdir(folder_path)
                      if os.path.isfile(os.path.join(folder_path, f))]
        conversations = build_dataset_from_single_files_windows(folder_path, train_files, window_size=3)

    else:

        file_path = "/home/shenxu/agent_failure/Automated_FA/Lib/experiment_outputs_Algorithm-Generated/train_files.json"
        with open(file_path, "r", encoding="utf-8") as f:
            train_files = json.load(f)
        file_path = "/home/shenxu/agent_failure/Automated_FA/Lib/experiment_outputs_Algorithm-Generated/test_files.json"
        with open(file_path, "r", encoding="utf-8") as f:
            test_files = json.load(f)
        conversations,all_labels, file_used = build_dataset_from_single_files(folder_path, train_files)

    # 2) toy 数据
    # conversations = [
    #     ["What is 2+2?", "It's 4.", "Yes, correct.", "We can move on."],
    #     ["Who discovered gravity?", "Newton did.", "He was an English scientist.", "Right."],
    #     ["What is the capital of France?", "Paris.", "Correct.", "It is in Europe."]
    # ]

    data = []
    for conv in conversations:
        emb = encoder.encode(conv)    # (seq_len, emb_dim)
        data.append(emb)

    emb_dim = data[0].size(-1)

    # 3) Qwen3-8B 作为解码器做重构（默认冻结主体，只训投影）
    model = QwenDecoderPredictor(
        emb_dim=emb_dim,
        base_model="/home/miaorui/project/Code/LLM/weight/meta-llama/Llama-3.1-8B-Instruct/",   # <- 按你环境中的实际可用权重名修改
        freeze_qwen=True,
        use_device_map_auto=True,
        torch_dtype=torch.bfloat16
    )

    # 如果没有使用 device_map="auto"，你可以把模型整体移到 device：
    # model.to(device)

    # 只训练投影层参数（若 freeze_qwen=True）
    optim_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(optim_params, lr=1e-5)

    # 4) 训练若干轮（teacher-forcing 重构）
    for epoch in range(10):
        loss = train_epoch_with_proto(model, data, optimizer, device=device)
        print(f"Epoch {epoch} | Recon Loss = {loss:.4f}")
        if loss<0.1:
            break
    ALL_SENTENCE = False
    if ALL_SENTENCE:
        test_seq, all_labels, file_used = build_dataset_from_single_all_sentence_files(folder_path, test_files, test_is=True)
    else:
    # 5) 推理：自回归 next-step reconstruction（逐步用“预测向量”扩展历史）
        test_seq, all_labels, file_used = build_dataset_from_single_files(folder_path, test_files,test_is=True)
    #test_seq = ["What is 5*6?", "It is 30.", "Correct.", "The moon is made of cheese."]
    #test_emb = encoder.encode(test_seq)      # (L, D)
    test_data = []
    for conv in test_seq:
        emb = encoder.encode(conv)  # (seq_len, emb_dim)
        test_data.append(emb)
    all_pred = []
    true_all = []
    all_score = []
    Teacher_forcing = True
    right_num = 0
    right_num_1 = 0
    TWO_C = True
    for i,test_emb in enumerate(test_data):
        if Teacher_forcing:
            #scores,sigmoid_score = anomaly_score_teacher_forcing(model, test_emb, device=device)
            scores, sigmoid_score, preds = anomaly_score_teacher_forcing_with_proto(model, test_emb, device=device,alpha = 0.9,beta=0.1)
            all_pred.append(preds)
            all_score.append(sigmoid_score)
            true_all.append(all_labels[i][1:])
        else:
            scores = anomaly_score_autoregressive_with_proto(model, test_emb, device=device)
        if ALL_SENTENCE:
            try:
                first_index = preds.index(1)
                print(f"元素1第一次出现的位置: {first_index}")
                if first_index == file_used[i]:
                    right_num += 1
            except ValueError:
                print("元素1不在列表中")
            # max_value = max(scores)
            # max_index = scores.index(max_value)
            # if max_index == file_used[i]:
            #     right_num+=1
            # if (max_index+1) == file_used[i]+1:
            #     right_num_1+=1
            # if scores[max_index] == file_used[i]:
            #     right_num+=1
            # if scores[max_index] == file_used[i]:
            #     right_num_1+=1
        # if ALL_SENTENCE:
        #     if TWO_C:
        #         s_all,p_all,s1,p1,scores,preds = anomaly_two_score_autoregressive_with_proto_pred(model,test_emb,device=device)
        #     else:
        #         scores,preds = anomaly_score_autoregressive_with_proto_pred(model, test_emb, device=device)
        # else:
        #
        #     scores = anomaly_score_autoregressive_with_proto(model, test_emb, device=device)
        #     scores = minmax_normalize(scores)
        # print("Autoregressive anomaly scores per step:", scores)
        # if ALL_SENTENCE:
        #     try:
        #         first_index = preds.index(1)
        #         print(f"元素1第一次出现的位置: {first_index}")
        #         if first_index == file_used[i]:
        #             right_num += 1
        #     except ValueError:
        #         print("元素1不在列表中")
        #     # max_value = max(scores)
        #     # max_index = scores.index(max_value)
        #     # if max_index == file_used[i]:
        #     #     right_num+=1
        #     # if (max_index+1) == file_used[i]+1:
        #     #     right_num_1+=1
        #     # if scores[max_index] == file_used[i]:
        #     #     right_num+=1
        #     # if scores[max_index] == file_used[i]:
        #     #     right_num_1+=1
        # else:
        #     max_value = max(scores)
        #     max_index = scores.index(max_value)
        #     if max_index == file_used[i]:
        #         right_num += 1
        #     if (max_index + 1) == file_used[i] + 1:
        #         right_num_1 += 1
        #     pred_list = [1 if p==1.0 else 0 for p in scores]
        #     all_score.append(scores)
        #     all_pred.append(pred_list)
        #     true_all.append(all_labels[i][1:])
    if ALL_SENTENCE:
        print(
            f"[Test Set] Acc: {right_num/len(file_used):.8f} ")
        print(
            f"[Test Set] Acc: {right_num_1 / len(file_used):.8f} ")
        flat_list_pred = [item for sublist in all_pred for item in sublist]
        flat_list_true = [item for sublist in true_all for item in sublist]
        flat_list_score = [item for sublist in all_score for item in sublist]
        from sklearn.metrics import roc_auc_score, average_precision_score
        import torch.nn.functional as F
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        precision, recall, f1, _ = precision_recall_fscore_support(flat_list_true, flat_list_pred, average='binary',
                                                                   zero_division=0)
        acc = accuracy_score(flat_list_true, flat_list_pred)
        AUROC = roc_auc_score(flat_list_true, flat_list_score)
        AUPRC = average_precision_score(flat_list_true, flat_list_score)
        print(
            f"[Test Set] Acc: {acc:.8f} | P: {precision:.8f} | R: {recall:.8f} | F1: {f1:.8f} | AUROC: {AUROC:.4f} | AUPRC: {AUPRC:.4f}")
    else:
        flat_list_pred = [item for sublist in all_pred for item in sublist]
        flat_list_true = [item for sublist in true_all for item in sublist]
        flat_list_score = [item for sublist in all_score for item in sublist]
        from sklearn.metrics import roc_auc_score, average_precision_score
        import torch.nn.functional as F
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(flat_list_true, flat_list_pred, average='binary', zero_division=0)
        acc = accuracy_score(flat_list_true, flat_list_pred)
        AUROC = roc_auc_score(flat_list_true, flat_list_score)
        AUPRC = average_precision_score(flat_list_true, flat_list_score)
        print(
            f"[Test Set] Acc: {acc:.8f} | P: {precision:.8f} | R: {recall:.8f} | F1: {f1:.8f} | AUROC: {AUROC:.4f} | AUPRC: {AUPRC:.4f}")
        print(
            f"[Test Set] Acc: {right_num / len(file_used):.8f} ")
        print(
            f"[Test Set] Acc: {right_num_1 / len(file_used):.8f} ")
