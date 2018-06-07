from keras.models import load_model
import numpy as np

generator = load_model('hw3-1/gan_model.h5')
filename = 'samples/gan.png'

def save_imgs(generator):
    import matplotlib.pyplot as plt
    r, c = 5, 5
    # noise = np.random.normal(0, 1, (r * c, 100))
    # np.save('gen_noise.npy',noise)
    noise = np.load('hw3-1/util/gen_noise.npy')
    # gen_imgs should be shape (25, 64, 64, 3)
    gen_imgs = generator.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,:])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig(filename)
    plt.close()

save_imgs(generator)