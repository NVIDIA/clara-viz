# clara-viz 0.1.1 (Dec 14 2021)

## Bug Fixes

* When installing the `clara-viz-core` Python package only there is the error `ModuleNotFoundError: No module named 'packaging'` when doing `import clara.viz.core`
* When getting the settings from the renderer the 'TransferFunctions' sections is returned as 'Transferfunctions' with lower case 'f'

## Documentation

* Added a section on using Clara Viz within a docker container.
* Added a link to the documentation.
* Added a section on WSL (Windows Subsystem for Linux).

## Notebooks

* The DataDefinition class is using ITK to load the data files, make sure ITK is available.
* Added a slice rendering example (Slice_rendering.ipynb)
* Fixed the check if the volume file exists in Render_image.ipynb, also fixed volume orientation and scaling.
* Updated the settings files to match the settings conventions used by the renderer.

## Misc

* Changed the camera names and removed the `Slice` prefix of the orthographic cameras. Renamed the perspective camera from `Cinematic` to `Perspective`

# clara-viz 0.1.0 (Dec 3 2021)

Initial release of Clara Viz
