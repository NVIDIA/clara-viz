# Clara Viz NvRTVol Render Server Example

Uses the NvRTVol renderer to visualize MHD/MHA files.

MHD/MHA is a file format used by ITK/VTK, see [ITK MetaIO](https://itk.org/Wiki/ITK/MetaIO)
for more information.

X11 is used to open a window and display the color buffer on the left side and the alpha
buffer in the middle and the depth buffer on the right side.

## Running

Start the sample with './bin/$(uname -m)/ClaraVizRenderer'.

Specify the file to load with the '-f' option. '-h' shows a help message.
