{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "toc_visible": true,
      "authorship_tag": "ABX9TyPZ6YXpKwWz5kd6q13D8dua",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Donjoyii/Google-Colaboratory/blob/main/Sar_processing_kmeanas.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "WrjNLiHfwgiK",
        "outputId": "8f856349-044f-461f-81e9-11bb34da5a71"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "hello goolge colaboratory\n",
            "Hello Sar\n"
          ]
        }
      ],
      "source": [
        "a = \"hello goolge colaboratory\"\n",
        "print(a)\n",
        "b = \"Hello Sar\"\n",
        "print(b)"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "a = plt.imread('river_2.bmp')\n",
        "a.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "167JuL31XKWw",
        "outputId": "daf5c089-7f1e-4a9f-dac2-7a689f71a543"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(256, 256)"
            ]
          },
          "metadata": {},
          "execution_count": 3
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# coding: utf-8\n",
        "import cv2\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "#读取原始图像灰度颜色\n",
        "img = cv2.imread('/content/Mosaic1_new_truth.tif', 0) \n",
        "print(img.shape)\n",
        "#获取图像高度、宽\n",
        "rows, cols = img.shape[:]\n",
        "#图像二维像素转换为一维\n",
        "data = img.reshape((rows * cols, 1))\n",
        "data = np.float32(data)\n",
        "#定义中心 (type,max_iter,epsilon)\n",
        "criteria = (cv2.TERM_CRITERIA_EPS +\n",
        "            cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)\n",
        "#设置标签\n",
        "flags = cv2.KMEANS_RANDOM_CENTERS\n",
        "#K-Means聚类 聚集成4类\n",
        "compactness, labels, centers = cv2.kmeans(data, 2, None, criteria, 10, flags)\n",
        "#生成最终图像\n",
        "dst = labels.reshape((img.shape[0], img.shape[1]))\n",
        "#用来正常显示中文标签\n",
        "plt.rcParams['font.sans-serif']=['SimHei']\n",
        "#显示图像\n",
        "titles = [u'initil', u'final'] \n",
        "images = [img, dst] \n",
        "for i in range(2): \n",
        " plt.subplot(1,2,i+1), plt.imshow(images[i], 'gray'), \n",
        " plt.title(titles[i]) \n",
        " plt.xticks([]),plt.yticks([]) \n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 221
        },
        "id": "xa6FSH-9UnPq",
        "outputId": "73ac5f62-8e64-4184-affa-41d5133102cf"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(256, 256)\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 2 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "rPm6FZjSbFVK"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}