import cv2
import mmcv
import numpy as np
import os
import torch
import matplotlib.pyplot as plt


def featuremap_2_heatmap(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    heatmap = feature_map[:,0,:,:]*0
    heatmaps = []
    for c in range(feature_map.shape[1]):
        heatmap+=feature_map[:,c,:,:]
    heatmap = heatmap.cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmaps.append(heatmap)

    return heatmaps


def draw_feature_map(features,save_dir = 'feature_map',name = None,i=0):
    img = mmcv.imread("../demo/IP00400004900270.jpg")
    if isinstance(features,torch.Tensor):
        for heat_maps in features:
            heat_maps=heat_maps.unsqueeze(0)
            heatmaps = featuremap_2_heatmap(heat_maps)
            # 这里的h,w指的是你想要把特征图resize成多大的尺寸
            heatmap = cv2.resize(heatmap, (256, 256))
            for heatmap in heatmaps:
                heatmap = np.uint8(255 * heatmap)
                # 下面这行将热力图转换为RGB格式 ，如果注释掉就是灰度图
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = heatmap
                plt.imshow(superimposed_img,cmap='gray')
                plt.show()
    else:
        for featuremap in features:
            heatmaps = featuremap_2_heatmap(featuremap)
            i=i+1
            for heatmap in heatmaps:
                heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
                heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
                heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
                superimposed_img = heatmap * 0.5 + img*0.3 # 这里的0.4是热力图强度因子
                # superimposed_img=heatmap
                plt.imshow(superimposed_img)
                plt.show()
                #plt.savefig(superimposed_img)
                #下面这些是对特征图进行保存，使用时取消注释
                # cv2.imshow("1",superimposed_img)
                # cv2.waitKey(0)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                print(os.path.join(save_dir,"tood+pafpn" +str(i)+'.png'))
                cv2.imwrite(os.path.join(save_dir,"tood+pafpn" +str(i)+'.png'), superimposed_img)
                # cv2.destroyAllWindows()
