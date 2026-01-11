% This is the code of the paper "A True Global Contrast Method for IR Small
% Target Detection under Complex Background", authored by Jinhui Han, Saed
% Moradi, Bo Zhou, Wei Wang, Qian Zhao and Zhen Luo, printed in IEEE TGRS,
% vol. xx, pp. xx, 2025.
clear all
close all
clc
DEBUG=1;

%% read image and show it
IMG_org=imread('./data/400.bmp');
IMG_org=double(rgb2gray(IMG_org));
IMG_org = mat2gray(IMG_org);
[M_org N_org]=size(IMG_org);
if DEBUG==1
    IMG_show=IMG_org;
    figure;
    imshow(IMG_show,[min(min(IMG_show)) max(max(IMG_show))]);
    title('IMG ORG')
end

%% set parameters used in IPI model
lambda=1/sqrt(max(N_org,M_org));
mu=10*lambda;
tol=1e-6;
max_iter = 1000;

opt.dw = 50;
opt.dh = 50;
opt.x_step = 10;
opt.y_step = 10;

x_start=1;
x_end=M_org-mod(M_org,opt.x_step);
y_start=1;
y_end=N_org-mod(N_org,opt.y_step);

kesi=0.1;
dilar=1;

%% Gaussian filtering
GAUS=[1 2 1; 2 4 2; 1 2 1]./16;
IMG_GAUS=zeros(M_org,N_org);
for i=2:1:M_org-1
    for j=2:1:N_org-1
        IMG_GAUS(i,j)=sum(sum(IMG_org(i-1:i+1,j-1:j+1).*GAUS));
    end
end
if DEBUG==1
    IMG_show=IMG_GAUS;
    figure;
    imshow(IMG_show,[min(min(IMG_show)) max(max(IMG_show))]);
    title('IMG GAUS')
end
% save image
IMG_sav=double(IMG_GAUS);
[MX,NX]=size(IMG_sav);
max1=max(max(IMG_sav));
min1=min(min(IMG_sav));
for i=1:1:MX
    for j=1:1:NX
        IMG_sav(i,j)=(round((IMG_sav(i,j)-min1)./(max1-min1).*255));
    end
end
IMG_sav=uint8(IMG_sav);
imwrite(IMG_sav,'./results/IMG_GAUS.tif','tiff');

%% Sparse and low rank decomposition using IPI algorithm
[IMG_L, IMG_S] = winRPCA_median(IMG_org, opt, mu, tol, max_iter);

if DEBUG==1
    IMG_show=IMG_L;
    figure;
    imshow(IMG_show,[min(min(IMG_show)) max(max(IMG_show))]);
    title('IMG L')
    IMG_show=IMG_S;
    figure;
    imshow(IMG_show,[min(min(IMG_show)) max(max(IMG_show))]);
    title('IMG S')
end
%% LMD for low rank part
IMG_L=max(0,IMG_L);
IMG_L_MAXDILA=IMG_L;
for i=dilar+1:1:M_org-dilar
    for j=dilar+1:1:N_org-dilar
        dilatmp=IMG_L(i-dilar:i+dilar,j-dilar:j+dilar);
        IMG_L_MAXDILA(i,j)=max(max(dilatmp));
    end
end
% show image
if DEBUG==1
    IMG_show=IMG_L_MAXDILA;
    figure;
    imshow(IMG_show,[min(min(IMG_show)) max(max(IMG_show))]);
    title('IMG L MAXDILA')    
end
% save image
IMG_sav=double(IMG_S);
[MX,NX]=size(IMG_sav);
max1=max(max(IMG_sav));
min1=min(min(IMG_sav));
for i=1:1:MX
    for j=1:1:NX
        IMG_sav(i,j)=(round((IMG_sav(i,j)-min1)./(max1-min1).*255));
    end
end
IMG_sav=uint8(IMG_sav);
imwrite(IMG_sav,'./results/IMG_T.tif','tiff');

IMG_sav=double(IMG_L);
[MX,NX]=size(IMG_sav);
max1=max(max(IMG_sav));
min1=min(min(IMG_sav));
for i=1:1:MX
    for j=1:1:NX
        IMG_sav(i,j)=(round((IMG_sav(i,j)-min1)./(max1-min1).*255));
    end
end
IMG_sav=uint8(IMG_sav);
imwrite(IMG_sav,'./results/IMG_B.tif','tiff');

IMG_sav=double(IMG_L_MAXDILA);
[MX,NX]=size(IMG_sav);
max1=max(max(IMG_sav));
min1=min(min(IMG_sav));
for i=1:1:MX
    for j=1:1:NX
        IMG_sav(i,j)=(round((IMG_sav(i,j)-min1)./(max1-min1).*255));
    end
end
IMG_sav=uint8(IMG_sav);
imwrite(IMG_sav,'./results/IMG_BLMD.tif','tiff');

%% TGCM calculation
TGCM=IMG_org;
for i=1:1:M_org
    for j=1:1:N_org
        TGCM(i,j)=max(1,IMG_GAUS(i,j)./max(kesi,IMG_L_MAXDILA(i,j))).*max(0,(IMG_GAUS(i,j)-IMG_L_MAXDILA(i,j)));
    end
end
% cut edge
TGCM_CUT=TGCM(x_start:x_end,y_start:y_end);

%% Weighting
IMG_S_NONNG=max(0,IMG_S);
TGCM=mat2gray(TGCM);
IMG_S_NONNG=mat2gray(IMG_S_NONNG);
TGCM_WEIGHT=TGCM.*IMG_S_NONNG;
% cut edge
TGCM_WEIGHT_CUT=TGCM_WEIGHT(x_start:x_end,y_start:y_end);
% show image
if DEBUG==1
    IMG_show=TGCM_CUT;
    figure;
    imshow(IMG_show,[min(min(IMG_show)) max(max(IMG_show))]);
    title('TGCM CUT')
    figure;
    mesh(IMG_show)
    saveas(gcf,'TGCM CUT 3D.tif','tiff');
end
if DEBUG==1
    IMG_show=TGCM_WEIGHT_CUT;
    figure;
    imshow(IMG_show,[min(min(IMG_show)) max(max(IMG_show))]);
    title('TGCM WEIGHT CUT')
    figure;
    mesh(IMG_show)
    saveas(gcf,'TGCM WEIGHT CUT 3D.tif','tiff');
end
% save image
IMG_sav=double(TGCM_CUT);
[MX,NX]=size(IMG_sav);
max1=max(max(IMG_sav));
min1=min(min(IMG_sav));
for i=1:1:MX
    for j=1:1:NX
        IMG_sav(i,j)=(round((IMG_sav(i,j)-min1)./(max1-min1).*255));
    end
end
IMG_sav=uint8(IMG_sav);
imwrite(IMG_sav,'./results/TGCM_CUT.tif','tiff');

IMG_sav=double(TGCM_WEIGHT_CUT);
[MX,NX]=size(IMG_sav);
max1=max(max(IMG_sav));
min1=min(min(IMG_sav));
for i=1:1:MX
    for j=1:1:NX
        IMG_sav(i,j)=(round((IMG_sav(i,j)-min1)./(max1-min1).*255));
    end
end
IMG_sav=uint8(IMG_sav);
imwrite(IMG_sav,'./results/TGCM_WEIGHT_CUT.tif','tiff');

%% ==== 计算 ROC 曲线（稳健版） ====
disp('开始计算ROC曲线...');

% 1) 选择分数图（你可能有 CUT 或未裁剪版本）
if exist('TGCM_WEIGHT_CUT', 'var')
    score_map = TGCM_WEIGHT_CUT;
else
    score_map = TGCM_WEIGHT;  % 如果你没裁边，就用未裁剪版本
end

% 2) 读取/准备真值掩膜 (必须与分数图同尺寸)
%    如果你自己已经把 GT_mask 加载到内存，就跳过 load 这行
load('./data/GT_mask.mat','GT_mask');

GT = logical(GT_mask);   % 确保是逻辑型

% 3) 如果分数图是裁剪过的，而 GT 还没裁剪，按相同索引裁剪 GT
if ~isequal(size(score_map), size(GT))
    if all(exist('x_start','var') & exist('x_end','var') & exist('y_start','var') & exist('y_end','var'))
        GT_try = GT(x_start:x_end, y_start:y_end);
        if isequal(size(score_map), size(GT_try))
            GT = GT_try;
        end
    end
end

% 4) 最终再检查一次尺寸是否一致
assert(isequal(size(score_map), size(GT)), ...
    sprintf('尺寸不一致：score_map=%s, GT=%s', mat2str(size(score_map)), mat2str(size(GT))));

% 5) 拉平成列向量，长度保证一致
s = score_map(:);
y = GT(:);  % 已是 logical

% 6) 扫描阈值并计算 Pd / Pfa
ths = linspace(min(s), max(s), 200);
tpr = zeros(size(ths));
fpr = zeros(size(ths));

for k = 1:numel(ths)
    T   = ths(k);
    pred = (s >= T);
    TP = sum(pred &  y);
    FP = sum(pred & ~y);
    FN = sum(~pred &  y);
    TN = sum(~pred & ~y);
    tpr(k) = TP / max(TP+FN, 1);
    fpr(k) = FP / max(FP+TN, 1);
end

% 7) AUC + 曲线
auc = trapz(fpr, tpr);
figure;
plot(fpr, tpr, 'b-', 'LineWidth', 2); hold on;
plot([0 1],[0 1],'k--');
grid on; xlabel('False Positive Rate (Pfa)'); ylabel('True Positive Rate (Pd)');
title(sprintf('ROC Curve (AUC = %.3f)', auc));
disp(['ROC计算完成，AUC = ', num2str(auc, '%.4f')]);


