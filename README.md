# TensorRT5_ShowLayer
Since TensorRT5's API has changed a lot compared to TensorRT4, especially the config class has been removed, and the trt_engine network layer generated after conversion cannot be printed. By reading each layer of the network in turn and judging the layer category, and outputting the layer's setting properties, you can achieve a better viewing effect.
