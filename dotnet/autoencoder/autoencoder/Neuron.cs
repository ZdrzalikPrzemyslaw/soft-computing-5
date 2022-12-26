namespace autoencoder;

public interface INeuron
{
    double Output(double[] input);
    void Update(double trainingStep, double expected, double output, double[] input);
}

public class LinearNeuron : INeuron
{
    private readonly double[] _weights;

    public LinearNeuron(List<double> weights)
    {
        _weights = weights.ToArray();
    }

    public double Bias { get; set; }


    public double Output(double[] input)
    {
        double sum = 0;
        for (var i = 0; i < _weights.Length; i++) sum += input[i] * _weights[i];

        sum += Bias;

        return sum;
    }

    public void Update(double trainingStep, double expected, double output, double[] input)
    {
        for (var i = 0; i < _weights.Length; i++) _weights[i] += trainingStep * (expected - output) * input[i];

        Bias = Bias == 0 ? 0 : Bias + (expected - output);
    }
}