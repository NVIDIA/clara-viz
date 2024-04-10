# Quick Start

## Installation

This will install all Clara Viz packages use pip:

```bash
$ pip install clara-viz
```

Clara Viz is using namespace packages. The main functionality is implemented in the 'clara-viz-core' package,
Jupyter Notebook widgets are found in the 'clara-viz-widgets' package.
So for example if you just need the renderer use

```bash
$ pip install clara-viz-core
```

## Use interactive widget in Jupyter Notebook

Install the Jupyter notebook widgets.

```bash
$ pip install clara-viz-widgets
```

Start Jupyter Lab, open the notebooks in the `notebooks` folder. Make sure Git LFS is installed when cloning the repo, else the data files are not downloaded correctly and you will see file open errors when using the example notebooks.

```python
from clara.viz.widgets import Widget
from clara.viz.core import Renderer
import numpy as np

# load a RAW CT data file (volume is 512x256x256 voxels)
input = np.fromfile("CT.raw", dtype=np.int16)
input = input.reshape((512, 256, 256))

display(Widget(Renderer(input)))
```

## Render CT data from Python

```python
from PIL import Image
import clara.viz.core
import numpy as np

# load a RAW CT data file (volume is 512x256x256 voxels)
input = np.fromfile("CT.raw", dtype=np.int16)
input = input.reshape((512, 256, 256))

# create the renderer
renderer = clara.viz.core.Renderer(input)

# render to a raw numpy array
output = renderer.render_image(1024, 768, image_type=clara.viz.core.ColorImageType.RAW_RGBA_U8)
rgb_data = np.asarray(output)[:, :, :3]

# show with PIL
image = Image.fromarray(rgb_data)
image.show()
```
