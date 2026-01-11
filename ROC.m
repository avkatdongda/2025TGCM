function [fpr, tpr, auc] = ROC(score_map, gt_mask, num_thresh)
    % ROC 计算并绘制 ROC 曲线
    if nargin < 3, num_thresh = 200; end
    s = score_map(:);
    y = gt_mask(:) > 0;
    ths = linspace(min(s), max(s), num_thresh);

    tpr = zeros(size(ths));
    fpr = zeros(size(ths));

    pos = sum(y==1); 
    neg = sum(y==0);

    for k = 1:num_thresh
        T = ths(k);
        pred = (s >= T);
        TP = sum(pred &  y);
        FP = sum(pred & ~y);
        FN = sum(~pred &  y);
        TN = sum(~pred & ~y);
        tpr(k) = TP / (TP+FN);
        fpr(k) = FP / (FP+TN);
    end

    auc = trapz(fpr, tpr);

    figure; plot(fpr, tpr, 'b-', 'LineWidth', 2); hold on;
    plot([0 1],[0 1],'k--');
    xlabel('False Positive Rate (Pfa)');
    ylabel('True Positive Rate (Pd)');
    title(sprintf('ROC Curve (AUC = %.3f)', auc));
    grid on;
end
