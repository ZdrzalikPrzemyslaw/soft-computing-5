namespace autoencoder;

public class Network
{
    private LinearLayer _inputLayer;
    private LinearLayer _hiddenLayer;
    private LinearLayer _outputLayer;


    public Network(int inputSize, int hiddenLayerSize, int minWeight, int maxWeight)
    {
        _inputLayer = new LinearLayer(inputSize, 1, minWeight, maxWeight, false);
        _hiddenLayer = new LinearLayer(hiddenLayerSize, inputSize, minWeight, maxWeight, false);
        _outputLayer = new LinearLayer(inputSize, hiddenLayerSize, minWeight, maxWeight, false);
    }


    public double[] quantization(double[] img)
    {
        List<List<double>> lista = new List<List<double>>();
        lista.Add(img.ToList());
        var inputLayerOutput = _inputLayer.output(lista);
        var hiddenLayerOutput = 
    }
    
}