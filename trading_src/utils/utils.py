import numpy as np

from sklearn.utils.class_weight import compute_class_weight


def reshape_as_image(x, img_width, img_height):
    """
    params:
    x: list of array(x = [[],[],...])
    img_width: width to set each element of x
    img_height: height to set each element of x
    
    size of each element of x must be multiplicatif of img_width * img_height
    """
    x_temp = np.zeros((len(x), img_height, img_width, 1))
    for i in range(x.shape[0]):
        x_temp[i] = np.reshape(x[i], (img_height, img_width, 1))

    return x_temp


def get_sample_weights(y):
    """
    calculate the sample weights based on class weights. Used for models with
    imbalanced data and one hot encoding prediction.

    params:
        y: class labels as integers
    """

    y = y.astype(int)  # compute_class_weight needs int labels
    class_weights = compute_class_weight('balanced', np.unique(y), y)
    
    print("real class weights are {}".format(class_weights), np.unique(y))
    print("value_counts", np.unique(y, return_counts=True))
    sample_weights = y.copy().astype(float)
    for i in np.unique(y):
        if i == 2:
            sample_weights[sample_weights == i] = class_weights[i]
        else: 
            class_weights[i] = class_weights[i] * 0.8
            sample_weights[sample_weights == i] = class_weights[i]
        # sample_weights = np.where(sample_weights == i, class_weights[int(i)], y_)

    return (sample_weights, class_weights)


#def show_images(rows, columns, path):
#    w = 15
#    h = 15
#    fig = plt.figure(figsize=(15, 15))
#    files = os.listdir(path)
#    for i in range(1, columns * rows + 1):
#        index = np.random.randint(len(files))
#        img = np.asarray(Image.open(os.path.join(path, files[index])))
#        fig.add_subplot(rows, columns, i)
#        plt.title(files[i], fontsize=10)
#        plt.subplots_adjust(wspace=0.5, hspace=0.5)
#        plt.imshow(img)
#    plt.show()