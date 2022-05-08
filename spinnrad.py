import matplotlib.pyplot as plt


def layerplot(img, model):
    """
    Visualize the layers of a convolution neural net.
    """

    model.eval()

    # Create list of layers
    layers=[]

    model_children=list(model.children())

    for child in model_children:
        for layer in child.children():
            layers.append(layer)

    # Plot input image (img)
    plt.figure(figsize=(2,2))
    plt.imshow(img.permute(1,2,0).data)
    plt.axis("off")
    plt.show()
    plt.close()

    # Run input image through layers
    img = img.unsqueeze(0)
    results = [layers[0](img)]
    for i in range(1, len(layers)):
        results.append(layers[i](results[-1]))
    outputs = results

    # Plot layers
    for num_layer in range(len(outputs)):
        plt.figure(figsize=(25, 20))
        if outputs[num_layer].dim()==4:
            layer_viz = outputs[num_layer][0, :, :, :]
        else:
            break
        layer_viz = layer_viz.data
        print("Layer",num_layer+1, layers[num_layer])
        for i, filter in enumerate(layer_viz):
            if i == 14: 
                break
            plt.subplot(1, 14, i + 1)
            plt.imshow(filter)
            plt.axis("off")
        plt.show()
        plt.close()
