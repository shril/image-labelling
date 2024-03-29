import labelling_tool

def flask_labeller(app, labelled_images, label_classes, config=None, use_reloader=True, debug=True):
    import json

    from flask import Flask, render_template, request, make_response, send_from_directory
    try:
        from flask_socketio import SocketIO, emit as socketio_emit
    except ImportError:
        SocketIO = None
        socketio_emit = None

    # Generate image IDs list
    image_ids = [str(i)   for i in range(len(labelled_images))]
    # Generate images table mapping image ID to image so we can get an image by ID
    images_table = {image_id: img   for image_id, img in zip(image_ids, labelled_images)}
    # Generate image descriptors list to hand over to the labelling tool
    # Each descriptor provides the image ID, the URL and the size
    image_descriptors = []
    for image_id, img in zip(image_ids, labelled_images):
        height, width = img.image_size
        image_descriptors.append(labelling_tool.image_descriptor(
            image_id=image_id, url='/image/{}'.format(image_id),
            width=width, height=height
        ))
    
    if SocketIO is not None:
        print('Using web sockets')
        socketio = SocketIO(app)
    else:
        socketio = None


    if config is None:
        config = {
            'tools': {
                'imageSelector': True,
                'labelClassSelector': True,
                'labelClassFilterInitial': label_class_filter_initial,
                'drawPolyLabel': True,
                'compositeLabel': True,
                'deleteLabel': True,
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


    @app.route('/')
    def index():
        label_classes_json = [cls.to_json()   for cls in label_classes]
        return render_template('labeller_page.jinja2',
                               tool_js_urls=labelling_tool.js_file_urls('/static/labelling_tool/'),
                               label_classes=json.dumps(label_classes_json),
                               image_descriptors=json.dumps(image_descriptors),
                               initial_image_index=0,
                               config=json.dumps(config),
                               use_websockets=socketio is not None)


    if socketio is not None:
        @socketio.on('get_labels')
        def handle_get_labels(arg_js):
            image_id = arg_js['image_id']

            image = images_table[image_id]

            labels, complete = image.get_label_data_for_tool()

            label_header = dict(labels=labels,
                                image_id=image_id,
                                complete=complete)

            socketio_emit('get_labels_reply', label_header)


        @socketio.on('set_labels')
        def handle_set_labels(arg_js):
            label_header = arg_js['label_header']

            image_id = label_header['image_id']

            image = images_table[image_id]

            image.set_label_data_from_tool(label_header['labels'], label_header['complete'])

            socketio_emit('set_labels_reply', '')


    else:
        @app.route('/labelling/get_labels/<image_id>')
        def get_labels(image_id):
            image = images_table[image_id]

            labels = image.labels_json
            complete = False


            label_header = {
                'labels': labels,
                'image_id': image_id,
                'complete': complete
            }

            r = make_response(json.dumps(label_header))
            r.mimetype = 'application/json'
            return r


        @app.route('/labelling/set_labels', methods=['POST'])
        def set_labels():
            label_header = json.loads(request.form['labels'])
            image_id = label_header['image_id']
            complete = label_header['complete']
            labels = label_header['labels']

            image = images_table[image_id]
            image.labels_json = labels

            return make_response('')


    @app.route('/image/<image_id>')
    def get_image(image_id):
        image = images_table[image_id]
        data, mimetype, width, height = image.data_and_mime_type_and_size()
        r = make_response(data)
        r.mimetype = mimetype
        return r



    @app.route('/ext_static/<path:filename>')
    def base_static(filename):
        return send_from_directory(app.root_path + '/../ext_static/', filename)


    # if socketio is not None:
    #     socketio.run(app, debug=debug, use_reloader=use_reloader)
    # else:
    #     app.run(debug=debug, use_reloader=use_reloader)

    return socketio

