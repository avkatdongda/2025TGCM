function [fpr, tpr, ths, auc] = roc_from_map(score_map, gt_mask, varargin)
% ROC_FROM_MAP  从检测分数图(越大越像目标)和真值掩膜生成ROC曲线
%
% [fpr, tpr, ths, auc] = roc_from_map(score_map, gt_mask, 'NumThresh', 201, 'EvalMask', [])
%
% 输入:
%   score_map : HxW 或 HxWxN 的分数图(双精度/单精度/uint都可)，数值越大越像目标
%   gt_mask   : HxW 或 HxWxN 的真值掩膜(逻辑/0-1)，1=目标, 0=背景
%
% 可选参数(Name-Value):
%   'NumThresh' : 阈值个数(默认 201)，从[min, max]等间距取
%   'EvalMask'  : 评估掩膜(同尺寸，逻辑型)。true的像素参与评估；若为空则全图参与
%
% 输出:
%   fpr : 1xT 虚警率(False Positive Rate)
%   tpr : 1xT 检测率(True Positive Rate, Pd)
%   ths : 1xT 对应阈值
%   auc : ROC曲线下面积(Area Under Curve)
%
% 说明:
%   - FPR/PFA 在像素级统计：FP / 背景像素数
%   - TPR/Pd 在像素级统计：TP / 目标像素数
%   - NaN 会被忽略(不参与评估)
%
% 作者: 你自己
% ---------------------------------------------------------------

p = inputParser;
p.addParameter('NumThresh', 201, @(x)isnumeric(x)&&isscalar(x)&&x>=2);
p.addParameter('EvalMask',  [], @(x)islogical(x) || isempty(x));
p.parse(varargin{:});
T = p.Results.NumThresh;
evalMask = p.Results.EvalMask;

% 基本检查
assert(isequal(size(score_map), size(gt_mask)), 'score_map 与 gt_mask 尺寸不一致');
score_map = double(score_map);
gt_mask   = logical(gt_mask);

% 统一到二维 (Npix x Nimg)
sz = size(score_map);
if numel(sz) == 2
    Nimg = 1;
    score_map = reshape(score_map, [], 1);
    gt_mask   = reshape(gt_mask,   [], 1);
    if isempty(evalMask)
        evalMask = true(size(gt_mask));
    else
        evalMask = reshape(logical(evalMask), [], 1);
    end
else
    Nimg = sz(3);
    score_map = reshape(score_map, [], Nimg);
    gt_mask   = reshape(gt_mask,   [], Nimg);
    if isempty(evalMask)
        evalMask = true(size(gt_mask));
    else
        evalMask = reshape(logical(evalMask), [], Nimg);
    end
end

% 忽略 NaN：在 evalMask 中剔除
nan_mask = ~isfinite(score_map);
evalMask = evalMask & ~nan_mask;

% 只保留参与评估的像素
s = score_map(evalMask);
y = gt_mask(evalMask);

% 正负样本数
pos = sum(y(:)==1);
neg = sum(y(:)==0);
assert(pos>0 && neg>0, '评估区域内需同时包含目标与背景像素');

% 阈值集合: 等间距 (也可改用分位数/唯一值以更精细)
minS = min(s); maxS = max(s);
if minS == maxS
    warning('score_map 在评估区域内为常数，ROC 无法区分；返回一条对角线。');
    fpr = [0 1]; tpr = [0 1]; ths = [maxS maxS]; auc = 0.5;
    return;
end
ths = linspace(minS, maxS, T);

tpr = zeros(1, T);
fpr = zeros(1, T);

% 扫阈值
for k = 1:T
    t = ths(k);
    pred = (s > t);             % 注意: >t，等于t 归为背景
    TP = sum(pred &  y);
    FP = sum(pred & ~y);
    % FN = sum(~pred &  y);
    % TN = sum(~pred & ~y);
    tpr(k) = TP / pos;
    fpr(k) = FP / neg;
end

% AUC (按FPR排序后梯形积分)
[fs, idx] = sort(fpr);
ts = tpr(idx);
auc = trapz(fs, ts);

% 绘图
figure; hold on;
plot(fpr, tpr, 'LineWidth', 2);
plot([0 1], [0 1], 'k--', 'LineWidth', 1); % 随机猜测参考线
xlim([0 1]); ylim([0 1]); grid on;
xlabel('False Positive Rate (Pfa)');
ylabel('True Positive Rate (Pd)');
title(sprintf('ROC (AUC = %.4f)', auc));
hold off;

end
