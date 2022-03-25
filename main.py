from urllib.request import urlopen

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans


def plot_data(data, x, y, xl, yl, title):
    plt.scatter(x=x, y=y, data=data)
    plt.title(title)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.show()


def show_elbow_plot(data, max_k):
    means = []
    inertias = []
    
    for k in range(1,max_k):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        means.append(k)
        inertias.append(kmeans.inertia_)
    
    plt.figure(figsize=(12,5))
    plt.plot(means, inertias, 'o-')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.show()


def plot_cluster(data, kmeans, x, y, c, xl, yl, title):
    plt.scatter(x=x, y=y, c=c, data=data)
    plt.scatter(
        kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
        s=250, marker="x", c="black", label="centers"
    )
    plt.title(title)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.legend(scatterpoints=1)


def main():
    source_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    download_filename = "iris.data"

    with urlopen(source_url) as response:
        content = response.read()
        with open(download_filename, "wb") as fo:
            fo.write(content.strip())

    processed_filename = "iris.csv"
    csv_header = b"sepal-length,sepal-width,petal-length,petal-width,class"

    with open(download_filename, "rb") as src_fo:
        with open(processed_filename, "wb") as fo:
            fo.write(csv_header)
            fo.write(b"\n")
            fo.write(src_fo.read())

    df = pd.read_csv(processed_filename)

    column_names = ["sepal-length", "petal-length"]
    selected_data = df[column_names]

    x, y = column_names
    plot_data(df, x, y, "Sepal Length", "Petal Length", "Selected Data")

    show_elbow_plot(selected_data, 10)

    k_min = 2
    k_max = 3
    cluster_columns = []

    for k in range(k_min, k_max + 1):
        kmeans = KMeans(k).fit(selected_data)
        df[f"cluster_{k}"] = kmeans.labels_
        cluster_columns.append(f"cluster_{k}")

    plt.figure(figsize=(12,8))
    for i, k_col in enumerate(cluster_columns, 1):
        plt.subplot(1, 2, i)
        x, y = column_names
        plot_cluster(df, kmeans, x, y, k_col, "Sepal Length", "Petal Length", k_col)
    plt.show()


if __name__ == "__main__":
    main()