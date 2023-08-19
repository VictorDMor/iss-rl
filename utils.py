import cv2
import imageio
# Render avi or gif


def render_frames(frame_array, savePath, fileName, fps, otype='AVI'):
    print('Creating replay ...', end=' ')
    if otype == 'AVI':
        fileName += '.avi'
        height, width, layers = frame_array[0].shape
        if layers == 1:
            layers = 0
        size = (width, height)
        out = cv2.VideoWriter(
            savePath + fileName, cv2.VideoWriter_fourcc(*'DIVX'), fps, size, layers)
        for i in range(len(frame_array)):
            out.write(frame_array[i])
        out.release()
        print('Done. Saved to {}'.format(savePath + fileName))
    else:
        print('Error: Invalid type, must be avi.')
