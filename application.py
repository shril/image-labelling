import click
import labelling_tool, flask_labeller
from flask import Flask, render_template, request, make_response, send_from_directory

app = Flask(__name__, static_folder='static')

@click.command()
@click.option('--slic', is_flag=True, default=False, help='Use SLIC segmentation to generate initial labels')
@click.option('--readonly', is_flag=True, default=False, help='Don\'t persist changes to disk')
def run_app(slic, readonly):
    
    label_classes = [
        labelling_tool.LabelClassGroup('Natural', [
            labelling_tool.LabelClass('tree', 'Trees', dict(default=[0, 255, 192], natural=[0, 255, 192],
                                                            artificial=[128, 128, 128])),
            labelling_tool.LabelClass('lake', 'Lake', dict(default=[0, 128, 255], natural=[0, 128, 255],
                                                           artificial=[128, 128, 128])),
        ]),
        labelling_tool.LabelClassGroup('Artificial', [
            labelling_tool.LabelClass('building', 'Buldings', dict(default=[255, 128, 0], natural=[128, 128, 128],
                                                                   artificial=[255, 128, 0])),
        ])]

    if slic:
        import glob
        from matplotlib import pyplot as plt
        from skimage.segmentation import slic as slic_segment

        labelled_images = []
        for path in glob.glob('images/*.jpg'):
            print('Segmenting {0}'.format(path))
            img = plt.imread(path)
            # slic_labels = slic_segment(img, 1000, compactness=20.0)
            slic_labels = slic_segment(img, 1000, slic_zero=True) + 1

            print('Converting SLIC labels to vector labels...')
            labels = labelling_tool.ImageLabels.from_label_image(slic_labels)

            limg = labelling_tool.LabelledImageFile(path, labels)
            labelled_images.append(limg)

        print('Segmented {0} images'.format(len(labelled_images)))
    else:
        # Load in .JPG images from the 'images' directory.
        labelled_images = labelling_tool.PersistentLabelledImage.for_directory('images', image_filename_pattern='*.jpg',
                                                                               readonly=readonly)
        print('Loaded {0} images'.format(len(labelled_images)))


    config = {
        'tools': {
            'imageSelector': True,
            'labelClassSelector': True,
            'drawPolyLabel': True,
            'compositeLabel': True,
            'deleteLabel': True,
            'colour_schemes': [dict(name='default', human_name='All'),
                               dict(name='natural', human_name='Natural'),
                               dict(name='artificial', human_name='Artifical')],
            'deleteConfig': {
                'typePermissions': {
                    'point': True,
                    'box': True,
                    'polygon': True,
                    'composite': True,
                    'group': True,
                }
            }
        }
    }

    result = flask_labeller.flask_labeller(app, labelled_images, label_classes, config=config)
    return result

resp = run_app()

if resp is not None:
    app = resp