
# Put all our constants here

#WORKSPACE_DIR      = '/home/h2amer/work/workspace/universality/'

WORKSPACE_DIR      = '../jpeg_inference_on_DNNs/'

models             = {  'IV1':         1,
                        'IV3':         2,
                        'IV4':         3,
                        'MobileNet':   4,
                        'MobileNetV2': 5,
                        'ResNet-V2-101': 6,
                        'Pnasnet_Large': 7,
                        'nasnet_mobile': 8,
                        'EfficientNet' : 9,
                        'InceptionResnetV2': 10,
                        'Vgg16'        : 11,
                        'Vgg19'        : 12

 }


# resnet_v2_50/predictions/Reshape_1:0 
final_tensor_names = {  'IV1':         'InceptionV1/Logits/Predictions/Reshape_1:0',
                        'IV3':         'softmax:0',
                        'IV4':         'InceptionV4/Logits/Predictions:0',
                        'MobileNet':   'MobilenetV1/Predictions/Reshape_1:0',
                        'MobileNetV2': 'MobilenetV2/Predictions/Reshape_1:0',
                        'ResNet-V2-101':'resnet_v2_101/predictions/Reshape_1:0',
                        'InceptionResnetV2': 'InceptionResnetV2/Logits/Predictions:0',
                        'Pnasnet_Large': 'final_layer/predictions:0',
                        'nasnet_mobile': 'final_layer/predictions:0',
                        'EfficientNet' : 'save_1/RestoreV2/shape_and_slices:0',
                        # 'Vgg16'        : 'Softmax:0',
                        'Vgg16'        : 'vgg_16/fc8/squeezed:0',
                        # 'Vgg19'        : 'Softmax:0'
                        'Vgg19'        : 'vgg_19/fc8/squeezed:0'

 }

resized_dimention = {  'IV1':            224,
                        'IV3':           299,
                        'IV4':           299,
                        'MobileNet':     224,
                        'MobileNetV2':   224,
                        'ResNet-V2-101': 299,
                        'InceptionResnetV2': 299,
                        'Pnasnet_Large': 331,
                        'nasnet_mobile': 224, 
                        'EfficientNet' : 224,
                        'Vgg16'        : 224,
                        # 'Vgg16'        : 299,
                        'Vgg19'        : 224,

 }

Frozen_Graph  =  {      'IV1':           'frozen_inception_v1_optimized.pb',
                        'IV3':           'classify_image_graph_def.pb',
                        'IV4':           'inception_v4_2016_09_09_frozen.pb',
                        'MobileNet':     'mobilenet_v1_1.0_224_frozen_optimized.pb',
                        'MobileNetV2':   'frozen_mobilenet_v2_optimized.pb',
                        'ResNet-V2-101': 'frozen_resnet_v2_101.pb',
                        'InceptionResnetV2': 'inception_resnet_v2_frozen.pb',
                        'Pnasnet_Large': 'frozen_pnasnet.pb', 
                        'nasnet_mobile': 'frozen_nasnet_mobile.pb',
                        # 'Vgg16'        : 'vgg16_frozen_graph.pb',
                        # 'Vgg16'        : 'frozen_vgg_16.pb',
                        'Vgg16'        : 'frozen_vgg_16_not_1.pb',
                        # 'Vgg19'        : 'vgg19_frozen_graph.pb',
                        'Vgg19'       : 'frozen_vgg_19.pb',
                        'EfficientNet' : 'efficientnet_model.pb'

                       
 }


##### All Models:
all_models             = [
                        'IV3',
                        'ResNet-V2-50',
                        'Vgg16',
                        'InceptionResnetV2',
                        'MobileNet', # next test
                        'IV1',
                        'IV4',
                        'ResNet-V2-101',
                        'Vgg19',
                        'EfficientNet',
                        'MobileNetV2',
                        'Pnasnet_Large',
                        'nasnet_mobile',
                        'alexNet'

]

examples_count = 50000

VGG_MEAN = [103.939, 116.779, 123.68]
vgg_synset_file_path = WORKSPACE_DIR + '/util/vgg_synset.txt'


QF_start = {
        
        '26' : 10,
        '28' : 20,
        '30' : 40, 
        '32' : 60 


}