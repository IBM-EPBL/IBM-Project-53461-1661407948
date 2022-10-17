@@ -0,0 +1,211 @@
{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "eOjSNtUdWswZ"
      },
      "outputs": [],
      "source": [
        "from tensorflow.keras.models import Sequential\n",
        "from tensorflow.keras.layers import Convolution2D,MaxPooling2D,Flatten,Dense\n",
        "from tensorflow.keras.preprocessing.image import ImageDataGenerator as idm\n",
        "import numpy as np\n",
        "import warnings\n",
        "#Supressing warnings\n",
        "warnings.filterwarnings('ignore')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Bh3asPkbXbrn",
        "outputId": "94d7c87c-aca9-4b15-a970-9ec83a474077"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Mounted at /content/drive\n"
          ]
        }
      ],
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "background_save": true,
          "base_uri": "https://localhost:8080/",
          "height": 469
        },
        "id": "KblaMbvsXetW",
        "outputId": "1952283f-4bd6-420e-ce03-f203ec6ce8eb"
      },
      "outputs": [
        {
          "ename": "SyntaxError",
          "evalue": "ignored",
          "output_type": "error",
          "traceback": [
            "\u001b[0;36m  File \u001b[0;32m\"<ipython-input-12-00a25734f87d>\"\u001b[0;36m, line \u001b[0;32m5\u001b[0m\n\u001b[0;31m    Xtrain = train_flowers.flow_from_directory('/content/Flowers-Dataset (2).zip'/,target_size=(76,76),class_mode='categorical',batch_size=100)\u001b[0m\n\u001b[0m                                                                                  ^\u001b[0m\n\u001b[0;31mSyntaxError\u001b[0m\u001b[0;31m:\u001b[0m invalid syntax\n"
          ]
        }
      ],
      "source": [
        "# Creating augmentation on training variable\n",
        "train_flowers=idm(rescale=1./255,zoom_range=0.2,horizontal_flip=True)\n",
        "\n",
        "# Passing training data to train variable\n",
        "Xtrain = train_flowers.flow_from_directory('/content/123128873_546b8b7355_n.jpg',target_size=(76,76),class_mode='categorical',batch_size=100)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "mGmbRioGXn23"
      },
      "outputs": [],
      "source": [
        "# Creating augmentation on testing variable\n",
        "test_flowers=idm(rescale=1./255)\n",
        "\n",
        "# Passing testing data to test variable\n",
        "Xtest = test_flowers.flow_from_directory('/content/drive/MyDrive/flower/Test',target_size=(76,76),class_mode='categorical',batch_size=100)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "vMnI02ycX3fD"
      },
      "outputs": [],
      "source": [
        "Flower_model = Sequential()\n",
        "Flower_model.add(Convolution2D(32,(3,3),activation='relu',input_shape=(76,76,3)))\n",
        "Flower_model.add(MaxPooling2D(pool_size=(2,2)))\n",
        "Flower_model.add(Flatten())\n",
        "Flower_model.add(Dense(300,activation='relu'))\n",
        "Flower_model.add(Dense(150,activation='relu'))\n",
        "Flower_model.add(Dense(5,activation='softmax'))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "C6l_S_v2X4mB"
      },
      "outputs": [],
      "source": [
        "Flower_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "ZUfb-e3cX_x9"
      },
      "outputs": [],
      "source": [
        "Flower_model.fit_generator(Xtrain,steps_per_epoch= len (Xtrain),epochs= 8,validation_data=Xtest,validation_steps= len (Xtest))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 165
        },
        "id": "5GxKRSmPYF9i",
        "outputId": "8e9e3a10-ba69-4f0f-f9e8-004050555e95"
      },
      "outputs": [
        {
          "ename": "NameError",
          "evalue": "ignored",
          "output_type": "error",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-8-1569fa3f58b3>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mFlower_model\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msave\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'Flower.h5'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
            "\u001b[0;31mNameError\u001b[0m: name 'Flower_model' is not defined"
          ]
        }
      ],
      "source": [
        "Flower_model.save('Flower.h5')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "4Y4EeCnbYJxw"
      },
      "outputs": [],
      "source": [
        "from tensorflow.keras.preprocessing import image"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 237
        },
        "id": "WlCUh93rYMUH",
        "outputId": "c8c71873-716b-4125-821f-e33886d1cefc"
      },
      "outputs": [
        {
          "ename": "NameError",
          "evalue": "ignored",
          "output_type": "error",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-7-30ddc2a48397>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mtest_img\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mimage\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mload_img\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'/content/8181477_8cb77d2e0f_n.jpg/content/drive/MyDrive/flower/Train/rose/8181477_8cb77d2e0f_n.jpg'\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mtarget_size\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m76\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m76\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      2\u001b[0m \u001b[0mtest_img\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mNameError\u001b[0m: name 'image' is not defined"
          ]
        }
      ],
      "source": [
        "test_img=image.load_img('/content/8181477_8cb77d2e0f_n.jpg/content/drive/MyDrive/flower/Train/rose/8181477_8cb77d2e0f_n.jpg',target_size=(76,76))\n",
        "test_img"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}