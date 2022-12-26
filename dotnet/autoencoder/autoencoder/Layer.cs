namespace autoencoder;

public class LinearLayer
{
    public List<INeuron> Neurons { get; }

    public LinearLayer(int neuronCount, int weightCount, double minWeight, double maxWeight, bool isBias)
    {
        IsBias = isBias;
        Neurons = new List<INeuron>();
        var r = new Random();
        var range = maxWeight - minWeight;
        for (var i = 0; i < neuronCount; i++)
        {
            List<double> weights = new();
            for (var j = 0; j < weightCount; j++) weights.Add(r.Next() * range + minWeight);
            INeuron neuron = new LinearNeuron(weights);
            Neurons.Add(neuron);
        }
    }

    public List<double> output(List<List<double>> input)
    {
        List<double> outp = new List<double>();
        for (var i = 0; i < Neurons.Count; i++)
        {
            outp.Add(Neurons[i].Output(input[i].ToArray()));
        }

        return outp;
    }

    public bool IsBias { get; }
}