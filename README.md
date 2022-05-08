# spinnrad

## Usage
1. Clone this repo.

2. Import **spinnrad** in your Python script.

```
import sys
sys.path.append('path/to/spinnrad')
from spinnrad import layerplot
```

3. Create a pytorch model `model` and a dataloader `dataloader`.

4. Get an example image and plot the layers.

```
img = next(iter(dataloader()))[0][0]
layerplot(img, model)
```

## Example Output
![image](https://user-images.githubusercontent.com/36032606/167302549-79210649-237f-42e6-b462-5c681fa80aef.png)
