const std::string kShowLayerType[24] = {
    "Convolution layer",
    "Fully connected layer",
    "Activation layer",
    "Pooling layer",
    "LRN layer",
    "Scale Layer",
    "SoftMax layer",
    "Deconvolution layer",
    "Concatenation layer",
    "Elementwise layer",
    "Plugin layer",
    "RNN Layer",
    "UnaryOp Operation Layer",
    "Padding Layer",
    "Shuffle Layer",
    "Reduce layer",
    "TopK Layer",
    "Gather Layer",
    "Matrix Multiply Layer",
    "Ragged softmax Layer",
    "Constant Layer",
    "RNNv2 layer",
    "Identity layer",
    "PluginV2 Layer"};
const std::string kShowActivationType[3] = {"kRELU", "kSIGMOID", "kTANH"};
const std::string kShowPoolingType[3] = {"kMAX", "kAVERAGE", "kMAX_AVERAGE_BLEND"};
const std::string kShowScaleType[3] = {"kUNIFORM", "kCHANNEL", "kELEMENTWISE"};
const std::string kShowElementWiseType[7] = {"kSUM", "kPROD", "kMAX", "kMIN", "kSUB", "kDIV", "kPOW"};
const std::string kShowUnaryType[6] = {"kEXP", "kLOG", "kSQRT", "kRECIP", "kABS", "kPOW"};
const std::string kShowReduceType[5] = {"kSUM", "kPROD", "kMAX", "kMIN", "kAVG"};
const std::string kShowMatrixOpType[3] = {"kNONE", "kTRANSPOSE", "kVECTOR"};
const std::string kShowDataType[4] = {"kFLOAT", "kHALF", "kINT8", "kINT32"};
 
void PrintLayerInfo(nvinfer1::INetworkDefinition *network)
{
    std::cout << "------Network layers------" << std::endl;
    int layer_num = network->getNbLayers();
    nvinfer1::ILayer *layer;
    int layer_index;
 
    for(int i=0; i<layer_num; i++)
    {
        layer = network->getLayer(i);
        layer_index  = static_cast<int>(layer->getType());
        switch(layer_index)
        {
        case 0:
        {
            nvinfer1::IConvolutionLayer *convlayer;
            convlayer = (nvinfer1::IConvolutionLayer*)layer;
            std::string showstring;
            showstring += std::to_string(i) + "--" + convlayer->getName() + "--";
            showstring += kShowLayerType[layer_index] + "--Settings: ";
            //in channels
            nvinfer1::ITensor *intensor = convlayer->getInput(0);
            showstring += "in_channels=" + std::to_string(intensor->getDimensions().d[0]) + ", ";
            //out channels
            nvinfer1::ITensor *outtensor = convlayer->getOutput(0);
            showstring += "out_channels=" + std::to_string(outtensor->getDimensions().d[0]) + ", ";
            //kernel size
            nvinfer1::DimsHW kdims = convlayer->getKernelSize();
            showstring += "kernel_size=[" + std::to_string(kdims.h()) + ", "
                    + std::to_string(kdims.w()) + "], ";
            //stride
            nvinfer1::DimsHW sdims = convlayer->getStride();
            showstring += "stride=[" + std::to_string(sdims.h()) + ", "
                    + std::to_string(sdims.w()) + "], ";
            //padding
            nvinfer1::DimsHW pdims = convlayer->getPadding();
            showstring += "padding=[" + std::to_string(pdims.h()) + ", "
                    + std::to_string(pdims.w()) + "], ";
            //dilation
            nvinfer1::DimsHW ddims = convlayer->getDilation();
            showstring += "dilation=[" + std::to_string(ddims.h()) + ", "
                    + std::to_string(ddims.w()) + "], ";
            //groups
            showstring += "groups=" + std::to_string(convlayer->getNbGroups()) + ", ";
            //bias
            nvinfer1::Weights bweight = convlayer->getBiasWeights();
            if(bweight.count>0)
            {
                showstring += "bias=True";
            }
            else
            {
                showstring += "bias=False";
            }
            std::cout << showstring << std::endl;
            break;
        }
        case 1:
        {
            nvinfer1::IFullyConnectedLayer *linearlayer;
            linearlayer = (nvinfer1::IFullyConnectedLayer*)layer;
            std::string showstring;
            showstring += std::to_string(i) + "--" + linearlayer->getName() + "--";
            showstring += kShowLayerType[layer_index] + "--Settings: ";
            //in features
            nvinfer1::ITensor *intensor = linearlayer->getInput(0);
            showstring += "in_features=" + std::to_string(intensor->getDimensions().d[0]) + ", ";
            //out channels
            nvinfer1::ITensor *outtensor = linearlayer->getOutput(0);
            showstring += "out_features=" + std::to_string(outtensor->getDimensions().d[0]) + ", ";
            //bias
            nvinfer1::Weights bweight = linearlayer->getBiasWeights();
            if(bweight.count>0)
            {
                showstring += "bias=True";
            }
            else
            {
                showstring += "bias=False";
            }
            std::cout << showstring << std::endl;
            break;
        }
        case 2:
        {
            nvinfer1::IActivationLayer *activatelayer;
            activatelayer = (nvinfer1::IActivationLayer*)layer;
            std::string showstring;
            showstring += std::to_string(i) + "--" + activatelayer->getName() + "--";
            showstring += kShowLayerType[layer_index] + "--Activation_Type=";
            int act_index = static_cast<int>(activatelayer->getActivationType());
            showstring += kShowActivationType[act_index];
            std::cout << showstring <<std::endl;
            break;
        }
        case 3:
        {
//            kernel_size, stride=None, padding=0, dilation=1,
//                          ceil_mode=False, return_indices=False):
            nvinfer1::IPoolingLayer *poolinglayer;
            poolinglayer = (nvinfer1::IPoolingLayer*)layer;
            std::string showstring;
            showstring += std::to_string(i) + "--" + poolinglayer->getName() + "--";
            //pooling type
            showstring += kShowLayerType[layer_index] + "--Pooling_Type=";
            int pooling_index = static_cast<int>(poolinglayer->getPoolingType());
            showstring += kShowPoolingType[pooling_index] +", Settings: ";
            //window size
            nvinfer1::DimsHW wdims = poolinglayer->getWindowSize();
            showstring += "window_size=[" + std::to_string(wdims.h()) + ", "
                    + std::to_string(wdims.w()) + "], ";
            //stride
            nvinfer1::DimsHW sdims = poolinglayer->getStride();
            showstring += "stride=[" + std::to_string(sdims.h()) + ", "
                    + std::to_string(sdims.w()) + "], ";
            //padding
            nvinfer1::DimsHW pdims = poolinglayer->getPadding();
            showstring += "padding=[" + std::to_string(pdims.h()) + ", "
                    + std::to_string(pdims.w()) + "]";
            std::cout << showstring << std::endl;
            break;
        }
        case 4:
        {
            nvinfer1::ILRNLayer *lrnlayer;
            lrnlayer = (nvinfer1::ILRNLayer*)layer;
            std::string showstring;
            showstring += std::to_string(i) + "--" + lrnlayer->getName();
            std::cout << showstring << std::endl;
            break;
        }
        case 5:
        {
 
            nvinfer1::IScaleLayer *scalelayer;
            scalelayer = (nvinfer1::IScaleLayer*)layer;
            std::string showstring;
            showstring += std::to_string(i) + "--" + scalelayer->getName() + "--";
            showstring += kShowLayerType[layer_index] + "--Scale_Type=";
            int scale_index = static_cast<int>(scalelayer->getMode());
            showstring += kShowScaleType[scale_index];
            std::cout << showstring <<std::endl;
            break;
        }
        case 6:
        {
            nvinfer1::ISoftMaxLayer *softmaxlayer;
            softmaxlayer = (nvinfer1::ISoftMaxLayer*)layer;
            std::string showstring;
            showstring += std::to_string(i) + "--" + softmaxlayer->getName() + "--";
            showstring += kShowLayerType[layer_index] + "--Settings: ";
            showstring += "axis=" + std::to_string(softmaxlayer->getAxes());
            std::cout << showstring << std::endl;
            break;
        }
        case 7:
        {
            nvinfer1::IDeconvolutionLayer *deconvlayer;
            deconvlayer = (nvinfer1::IDeconvolutionLayer*)layer;
            std::string showstring;
            showstring += std::to_string(i) + "--" + deconvlayer->getName() + "--";
            showstring += kShowLayerType[layer_index] + "--Settings: ";
            //in channels
            nvinfer1::ITensor *intensor = deconvlayer->getInput(0);
            showstring += "in_channels=" + std::to_string(intensor->getDimensions().d[0]) + ", ";
            //out channels
            nvinfer1::ITensor *outtensor = deconvlayer->getOutput(0);
            showstring += "out_channels=" + std::to_string(outtensor->getDimensions().d[0]) + ", ";
            //kernel size
            nvinfer1::DimsHW kdims = deconvlayer->getKernelSize();
            showstring += "kernel_size=[" + std::to_string(kdims.h()) + ", "
                    + std::to_string(kdims.w()) + "], ";
            //stride
            nvinfer1::DimsHW sdims = deconvlayer->getStride();
            showstring += "stride=[" + std::to_string(sdims.h()) + ", "
                    + std::to_string(sdims.w()) + "], ";
            //padding
            nvinfer1::DimsHW pdims = deconvlayer->getPadding();
            showstring += "padding=[" + std::to_string(pdims.h()) + ", "
                    + std::to_string(pdims.w()) + "], ";
            //groups
            showstring += "groups=" + std::to_string(deconvlayer->getNbGroups()) + ", ";
            //bias
            nvinfer1::Weights bweight = deconvlayer->getBiasWeights();
            if(bweight.count>0)
            {
                showstring += "bias=True";
            }
            else
            {
                showstring += "bias=False";
            }
            std::cout << showstring << std::endl;
            break;
        }
        case 8:
        {
            nvinfer1::IConcatenationLayer *concatlayer;
            concatlayer = (nvinfer1::IConcatenationLayer*)layer;
            std::string showstring;
            showstring += std::to_string(i) + "--" + concatlayer->getName() + "--";
            showstring += kShowLayerType[layer_index] + "--Settings: ";
            showstring += "axis=" + std::to_string(concatlayer->getAxis());
            std::cout << showstring << std::endl;
            break;
        }
        case 9:
        {
            nvinfer1::IElementWiseLayer *elementlayer;
            elementlayer = (nvinfer1::IElementWiseLayer*)layer;
            std::string showstring;
            showstring += std::to_string(i) + "--" + elementlayer->getName() + "--";
            showstring += kShowLayerType[layer_index] + "--Elementwise_Type=";
            int elem_index = static_cast<int>(elementlayer->getOperation());
            showstring += kShowElementWiseType[elem_index];
            std::cout << showstring <<std::endl;
            break;
        }
        case 10:
        {
            nvinfer1::IPluginLayer *pluginlayer;
            pluginlayer = (nvinfer1::IPluginLayer*)layer;
            std::string showstring;
            showstring += std::to_string(i) + "--" + pluginlayer->getName() + "--";
            showstring += kShowLayerType[layer_index];
            std::cout << showstring << std::endl;
            break;
        }
        case 11:
        {
            nvinfer1::IRNNLayer *rnnlayer;
            rnnlayer = (nvinfer1::IRNNLayer*)layer;
            std::string showstring;
            showstring += std::to_string(i) + "--" + rnnlayer->getName() + "--";
            showstring += kShowLayerType[layer_index];
            std::cout << showstring << std::endl;
            break;
        }
        case 12:
        {
            nvinfer1::IUnaryLayer *unarylayer;
            unarylayer = (nvinfer1::IUnaryLayer*)layer;
            std::string showstring;
            showstring += std::to_string(i) + "--" + unarylayer->getName() + "--";
            showstring += kShowLayerType[layer_index] + "--Unary_Type=";
            int unary_index = static_cast<int>(unarylayer->getOperation());
            showstring += kShowUnaryType[unary_index];
            std::cout << showstring <<std::endl;
            break;
        }
        case 13:
        {
            nvinfer1::IPaddingLayer *padlayer;
            padlayer = (nvinfer1::IPaddingLayer*)layer;
            std::string showstring;
            showstring += std::to_string(i) + "--" + padlayer->getName() + "--";
            showstring += kShowLayerType[layer_index];
            std::cout << showstring << std::endl;
            break;
        }
        case 14:
        {
            //layer index, layer name, layer type
            nvinfer1::IShuffleLayer *shufflelayer;
            shufflelayer = (nvinfer1::IShuffleLayer*)layer;
            std::string showstring;
            showstring += std::to_string(i) + "--" + shufflelayer->getName() + "--";
            showstring += kShowLayerType[layer_index] + "--";
            //first transpose order
            int *porder = shufflelayer->getFirstTranspose().order;
            int order[3] = {porder[0], porder[1], porder[2]};
            if(!(order[0]==0 && order[1]==1 && order[2]==2))
            {
                showstring += "Shuffle_Type=kTRANSPOSE0--Settings: order=[";
                showstring += std::to_string(order[0]) + ", " + std::to_string(order[1]) + ", " +
                        std::to_string(order[2]) + "]";
            }
            //second transpose order
            porder = shufflelayer->getSecondTranspose().order;
            order[0] = porder[0];
            order[1] = porder[1];
            order[2] = porder[2];
            if(order[0]!=0 || order[1] !=1 || order[2]!=2)
            {
                showstring += "Shuffle_Type=kTRANSPOSE1--Settings: order=[";
                showstring += std::to_string(order[0]) + ", " + std::to_string(order[1]) + ", " +
                        std::to_string(order[2]) + "]";
            }
            //reshape dims
            nvinfer1::Dims rdims = shufflelayer->getReshapeDimensions();
            if(rdims.nbDims==3)
            {
                if(rdims.d[0]!=0 || rdims.d[1]!=0 || rdims.d[2]!=0)
                {
                    showstring += "Shuffle_Type=kRESHAPE--Settings: dims=[";
                    for(int j=0; j<rdims.nbDims; j++)
                    {
                        showstring += std::to_string(rdims.d[j]) + ", ";
                    }
                    showstring += "]";
                }
            }
            std::cout << showstring << std::endl;
            break;
        }
        case 15:
        {
            nvinfer1::IReduceLayer *reducelayer;
            reducelayer = (nvinfer1::IReduceLayer*)layer;
            std::string showstring;
            showstring += std::to_string(i) + "--" + reducelayer->getName() + "--";
            showstring += kShowLayerType[layer_index] + "--Reduce_Type=";
            int reduce_index = static_cast<int>(reducelayer->getOperation());
            showstring += kShowReduceType[reduce_index] + "--Settings: axis=";
            showstring += std::to_string(reducelayer->getReduceAxes()) + ", keep_dims=";
            showstring += std::to_string(reducelayer->getKeepDimensions());
            std::cout << showstring << std::endl;
            break;
        }
        case 16:
        {
            nvinfer1::ITopKLayer *topklayer;
            topklayer = (nvinfer1::ITopKLayer*)layer;
            std::string showstring;
            showstring += std::to_string(i) + "--" + topklayer->getName() + "--";
            showstring += kShowLayerType[layer_index] + "--Settings: axis=";
            showstring += std::to_string(topklayer->getReduceAxes()) + ", k=";
            showstring += std::to_string(topklayer->getK());
            std::cout << showstring << std::endl;
            break;
        }
        case 17:
        {
            nvinfer1::IGatherLayer *gatherlayer;
            gatherlayer = (nvinfer1::IGatherLayer*)layer;
            std::string showstring;
            showstring += std::to_string(i) + "--" + gatherlayer->getName() + "--";
            showstring += kShowLayerType[layer_index] + "--Settings: axis=";
            showstring += std::to_string(gatherlayer->getGatherAxis());
            std::cout << showstring << std::endl;
            break;
        }
        case 18:
        {
            nvinfer1::IMatrixMultiplyLayer *matmullayer;
            matmullayer = (nvinfer1::IMatrixMultiplyLayer*)layer;
            std::string showstring;
            showstring += std::to_string(i) + "--" + matmullayer->getName() + "--";
            showstring += kShowLayerType[layer_index] + "--Settings: A_op=";
            int aop_index = static_cast<int>(matmullayer->getOperation(0));
            showstring += kShowMatrixOpType[aop_index] + ", A_transpose=";
            showstring += std::to_string(matmullayer->getTranspose(0)) + ", B_op=";
            int bop_index = static_cast<int>(matmullayer->getOperation(1));
            showstring += kShowMatrixOpType[bop_index] + ", B_transpose=";
            showstring += std::to_string(matmullayer->getTranspose(1));
            std::cout << showstring << std::endl;
            break;
        }
        case 19:
        {
            nvinfer1::IRaggedSoftMaxLayer *ragsofxmaxlayer;
            ragsofxmaxlayer = (nvinfer1::IRaggedSoftMaxLayer*)layer;
            std::string showstring;
            showstring += std::to_string(i) + "--" + ragsofxmaxlayer->getName() + "--";
            showstring += kShowLayerType[layer_index];
            std::cout << showstring << std::endl;
            break;
        }
        case 20:
        {
            //layer index, layer name, layer type
            nvinfer1::IConstantLayer *constlayer;
            constlayer = (nvinfer1::IConstantLayer*)layer;
            std::string showstring;
            showstring += std::to_string(i) + "--" + constlayer->getName() + "--";
            showstring += kShowLayerType[layer_index] + "--Settings: dims=[";
            nvinfer1::Dims cdims = constlayer->getDimensions();
            for(int j=0; j<cdims.nbDims; j++)
            {
                showstring += std::to_string(cdims.d[i]) + ", ";
            }
            showstring += "]";
            std::cout << showstring << std::endl;
            break;
        }
        case 21:
        {
            nvinfer1::IRNNv2Layer *rnnv2layer;
            rnnv2layer = (nvinfer1::IRNNv2Layer*)layer;
            std::string showstring;
            showstring += std::to_string(i) + "--" + rnnv2layer->getName() + "--";
            showstring += kShowLayerType[layer_index];
            std::cout << showstring << std::endl;
            break;
        }
        case 22:
        {
            nvinfer1::IIdentityLayer *identitylayer;
            identitylayer = (nvinfer1::IIdentityLayer*)layer;
            std::string showstring;
            showstring += std::to_string(i) + "--" + identitylayer->getName() + "--";
            showstring += kShowLayerType[layer_index] + "--Output_Datatype=";
            int dt_index = static_cast<int>(identitylayer->getOutputType(0));
            showstring += kShowDataType[dt_index];
            std::cout << showstring << std::endl;
            break;
        }
        case 23:
        {
            nvinfer1::IPluginV2Layer *pluginv2layer;
            pluginv2layer = (nvinfer1::IPluginV2Layer*)layer;
            std::string showstring;
            showstring += std::to_string(i) + "--" + pluginv2layer->getName() + "--";
            showstring += kShowLayerType[layer_index];
            std::cout << showstring << std::endl;
            break;
        }
        default:
            std::cout << i << "--nontype layer" << std::endl;
            break;
        }
    }
}
