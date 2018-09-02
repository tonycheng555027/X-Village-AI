from sklearn import cluster, datasets, metrics
import matplotlib.pyplot as plt
import time
# 讀入鳶尾花資料
iris = datasets.load_iris()
iris_X = iris.data
# # KMeans 演算法
# kmeans_fit = cluster.KMeans(n_clusters = 3).fit(iris_X)

# #印出分群結果
# cluster_labels = kmeans_fit.labels_
# print("分群結果：")
# print(cluster_labels)
# print("---")

# # 印出品種看看
# iris_y = iris.target
# print("真實品種：")
# print(iris_y)

# 迴圈
silhouette_avgs = []
ks = range(2, 11)
ini = range(10, 70, 20)
a = ['auto', 'full', 'elkan']

def Km(ks, ini, a):
    start_time = time.time()
    kmeans_fit = cluster.KMeans(n_clusters = ks, n_init = ini, algorithm = a).fit(iris_X)
    cluster_labels = kmeans_fit.labels_
    print(cluster_labels)
    silhouette_avg = metrics.silhouette_score(iris_X, cluster_labels)
    silhouette_avgs.append(silhouette_avg)
    end_time = time.time()
    print('Time elapsed:\t', end_time - start_time)

for i in ks:
    Km(i, 10, 'auto')
# for j in ini:
#     Km(3, j, 'auto')
# for k in range(3):
#     Km(3, 10, a[k])
#     print(a[k])

print(silhouette_avgs)
# 作圖並印出 k = 2 到 10 的績效
plt.bar(ks, silhouette_avgs)
plt.show()
