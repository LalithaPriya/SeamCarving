import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


def imshow(im, im_title=None, shape=None, interpolation='bilinear', cmap=None, sup_title='Figure(s)', slider_attr=None,
           slider_callback=None,
           button_attr=None, button_callback=None):
    """
    My favorite method that abstracts all the matplotlib code when building an interactive canvas that have as many
    sliders and buttons which can alter what is shown on the plots
    This method opens a new window that displays image(s) using matplotlib. Optionally the methods accepts params that
    define interactive sliders and buttons in the canvas
    :param im: single image or list of images (greyscale or multi channel)
    :param im_title: single string or list of strings defining the title for each image in $im
    :param shape: shape of subplots eg.: (3,1) makes fig to have 3 subplots one above the other
    :param interpolation: interpolation logic eg.: 'nearest', 'bicubic', 'bilinear', ..
    :param cmap: color map of the plot
    :param sup_title: main title of the entire figure
    :param slider_attr: a list of set of slider attributes. one set for each slider that defines valmax, valmin, valint
    :param slider_callback: a list of callback methods each accepting the params {event, axs, sliders, buttons} which
    define the state of figure/canvas
    :param button_attr: a list of set of button attributes. one for each slider that defines {'label'}
    :param button_callback: a list of callback methods each accepting the params {event, axs, sliders, buttons} which
    define the state of figure/canvas
    :return: None
    """
    # TODO : This method has become bulky. Work on simplification.
    if not type(im) == list:
        im = [im]
    # determine squarish shape to arrange all images
    shape = [shape, (int(np.sqrt(len(im))), int(np.ceil(len(im) / int(np.sqrt(len(im))))))][shape is None]
    im_title = [im_title, 'Image'][im_title is None]
    fig, axs = plt.subplots(shape[0], shape[1])
    if not type(axs) == list:
        axs = np.asarray([axs])
    axs = axs.flatten()
    # make ticks and axes(axis) disappear
    for ax in axs:
        ax.set_axis_off()
    # plot each image in its axes
    for i in range(len(im)):
        if cmap is not None:
            axs[i].imshow(im[i], interpolation=interpolation, cmap=cmap)
        else:
            axs[i].imshow(im[i], interpolation=interpolation)
        axs_title = '%s $%sx%s$\n$mn=%.3f$  $mx=%.3f$ ' % (
            im_title if not type(im_title) == list else im_title[i], im[i].shape[0], im[i].shape[1], np.min(im[i]),
            np.max(im[i]))
        axs[i].set_title(axs_title, fontweight='bold')
        axs[i].set_axis_off()
    # create widgets to interact with images
    num_sliders = 0 if slider_attr is None else len(slider_attr)
    num_buttons = 0 if button_attr is None else len(button_attr)

    widget_width = 0.05
    fig.subplots_adjust(bottom=widget_width * (num_buttons + num_sliders + 1))

    def create_slider(i):
        slider_ax = fig.add_axes([0.2, widget_width * (num_sliders - i), 0.65, 0.03], facecolor='grey')

        slider = Slider(slider_ax, slider_attr[i].get('label', '%s' % i), slider_attr[i].get('valmin', 0),
                        slider_attr[i].get('valmax', 1),
                        slider_attr[i].get('valint', slider_attr[i].get('valmin', 0)), color='#6baeff')
        slider.on_changed(lambda x: update_images(x, slider_callback[i]))
        return slider

    def create_button(i):
        button_ax = fig.add_axes([0.75, widget_width * (num_sliders) + widget_width * (num_buttons - i), 0.1, 0.03],
                                 facecolor='grey')
        button = Button(button_ax, button_attr[i].get('label', '%s' % i), color='0.99', hovercolor='0.575')
        button.on_clicked(lambda event: update_images(event, button_callback[i]))
        return button

    # create sliders and store them in memory
    sliders = list(map(create_slider, range(num_sliders)))
    # create buttons and store them in memory
    buttons = list(map(create_button, range(num_buttons)))

    # method that is called when a slider or button is touched. This method in turn
    # calls the callbacks to get the updated images and put them in the plot
    def update_images(event, callback):
        updates = callback(event, axs, sliders, buttons)
        if updates is not None and type(updates) == tuple and len(updates) > 0:
            updated_i_s = updates[0]
            updated_im_s = updates[1]
            for u_i, u_im in zip(updated_i_s, updated_im_s):
                if u_i < len(im):
                    if cmap is not None:
                        axs[u_i].imshow(u_im, interpolation=interpolation, cmap=cmap)
                    else:
                        axs[u_i].imshow(u_im, interpolation=interpolation)

    # set main title
    fig.canvas.manager.set_window_title(sup_title)
    plt.suptitle(sup_title)
    # bigger viewing area
    fig.set_size_inches(2 * fig.get_size_inches())
    plt.show()

