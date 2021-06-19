
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
dict1 = {19: '11-large_omnivores_and_herbivores', 29: '15-reptiles', 0: '4-fruit_and_vegetables', 11: '14-people', 1: '1-fish', 86: '5-household_electrical_devices', 90: '18-vehicles_1', 28: '3-food_containers', 23: '10-large_natural_outdoor_scenes', 31: '11-large_omnivores_and_herbivores', 39: '5-household_electrical_devices', 96: '17-trees', 82: '2-flowers', 17: '9-large_man-made_outdoor_things', 71: '10-large_natural_outdoor_scenes', 8: '18-vehicles_1', 97: '8-large_carnivores', 80: '16-small_mammals', 74: '16-small_mammals', 59: '17-trees', 70: '2-flowers', 87: '5-household_electrical_devices', 84: '6-household_furniture', 64: '12-medium_mammals', 52: '17-trees', 42: '8-large_carnivores', 47: '17-trees', 65: '16-small_mammals', 21: '11-large_omnivores_and_herbivores', 22: '5-household_electrical_devices', 81: '19-vehicles_2', 24: '7-insects', 78: '15-reptiles', 45: '13-non-insect_invertebrates', 49: '10-large_natural_outdoor_scenes', 56: '17-trees', 76: '9-large_man-made_outdoor_things', 89: '19-vehicles_2', 73: '1-fish', 14: '7-insects', 9: '3-food_containers', 6: '7-insects', 20: '6-household_furniture', 98: '14-people', 36: '16-small_mammals', 55: '0-aquatic_mammals', 72: '0-aquatic_mammals', 43: '8-large_carnivores', 51: '4-fruit_and_vegetables', 35: '14-people', 83: '4-fruit_and_vegetables', 33: '10-large_natural_outdoor_scenes', 27: '15-reptiles', 53: '4-fruit_and_vegetables', 92: '2-flowers', 50: '16-small_mammals', 15: '11-large_omnivores_and_herbivores', 18: '7-insects', 46: '14-people', 75: '12-medium_mammals', 38: '11-large_omnivores_and_herbivores', 66: '12-medium_mammals', 77: '13-non-insect_invertebrates', 69: '19-vehicles_2', 95: '0-aquatic_mammals', 99: '13-non-insect_invertebrates', 93: '15-reptiles', 4: '0-aquatic_mammals', 61: '3-food_containers', 94: '6-household_furniture', 68: '9-large_man-made_outdoor_things', 34: '12-medium_mammals', 32: '1-fish', 88: '8-large_carnivores', 67: '1-fish', 30: '0-aquatic_mammals', 62: '2-flowers', 63: '12-medium_mammals', 40: '5-household_electrical_devices', 26: '13-non-insect_invertebrates', 48: '18-vehicles_1', 79: '13-non-insect_invertebrates', 85: '19-vehicles_2', 54: '2-flowers', 44: '15-reptiles', 7: '7-insects', 12: '9-large_man-made_outdoor_things', 2: '14-people', 41: '19-vehicles_2', 37: '9-large_man-made_outdoor_things', 13: '18-vehicles_1', 25: '6-household_furniture', 10: '3-food_containers', 57: '4-fruit_and_vegetables', 5: '6-household_furniture', 60: '10-large_natural_outdoor_scenes', 91: '1-fish', 3: '8-large_carnivores', 58: '18-vehicles_1', 16: '3-food_containers'}
ansdict = {"fish": "鱼", "large_omnivores_and_herbivores": "大型杂食动物", "flowers": "花卉", "aquatic_mammals": "水生哺乳动物", "food_containers": "食品容器", "fruit_and_vegetables": "水果", "household_electrical_devices": "家用电器", "household_furniture": "家具", "insects": "昆虫", "large_carnivores": "大型肉食动物", "large_man-made_outdoor_things": "建筑", "large_natural_outdoor_scenes": "自然景色", "medium_mammals": "中型哺乳动物", "non-insect_invertebrates": "非昆虫无脊椎动物", "people": "人物", "reptiles": "爬行动物", "small_mammals": "小型哺乳动物", "trees": "树木", "vehicles_1": "交通工具", "vehicles_2": "交通工具"}

def getBaseData():
    (x_img_train, y_label_train), (x_img_test, y_label_test) = tf.keras.datasets.cifar100.load_data()
    mean = np.mean(x_img_train, axis=(0, 1, 2, 3))  # 四个维度 批数 像素x像素
    std = np.std(x_img_train, axis=(0, 1, 2, 3))
    return mean, std
def mymodel(path):
    model = tf.keras.Sequential()
    # conv1
    model.add(
        tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], input_shape=(32, 32, 3), strides=1, activation='relu',
                               padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
    model.add(tf.keras.layers.BatchNormalization())  # 批标准化
    model.add(tf.keras.layers.Dropout(0.3))  # 随机丢弃神经元，防止过拟合
    # conv2
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=1, activation='relu',
                                     padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
    model.add(tf.keras.layers.BatchNormalization())
    # 最大池化1
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # conv3
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], strides=1, activation='relu',
                                     padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.4))
    # conv4
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], strides=1, activation='relu',
                                     padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
    model.add(tf.keras.layers.BatchNormalization())
    # 最大池化2
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # conv5
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=1, activation='relu',
                                     padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.4))
    # conv6
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=1, activation='relu',
                                     padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.4))
    # conv7
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=1, activation='relu',
                                     padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
    model.add(tf.keras.layers.BatchNormalization())
    # 最大池化3
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # conv8
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=1, activation='relu',
                                     padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.4))
    # conv9
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=1, activation='relu',
                                     padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.4))
    # conv10
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=1, activation='relu',
                                     padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
    model.add(tf.keras.layers.BatchNormalization())
    # 最大池化4
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # conv11
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=1, activation='relu',
                                     padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.4))
    # conv12
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=1, activation='relu',
                                     padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.4))
    # conv13
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=1, activation='relu',
                                     padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
    model.add(tf.keras.layers.BatchNormalization())
    # 最大池化5
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # 全连接 MLP三层
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(rate=0.5))

    model.add(tf.keras.layers.Dense(units=512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(units=100))
    model.add(tf.keras.layers.Activation('softmax'))
    # 查看摘要
    # model.summary()
    model.load_weights(path)
    return model
mean,std = getBaseData()
def predict_img(filepath, model):
    img = image.load_img(filepath, target_size=(32, 32))
    img = image.img_to_array(img)
    img = img.reshape(1, 32, 32,3)
    img = (img-mean)/(std+1e-7)
    x = model.predict(img)
    _,index = np.where(x==x.max())
    strindex = dict1[index[0]]
    strindex = strindex.split('-',1)[1]
    print(ansdict[strindex])

if __name__ == '__main__':
    model = mymodel('my_train_model1.h5')
    childfile = os.listdir()
    for child in childfile:
        if(child.split('.')[-1]=="png"):
            predict_img(child,model)
