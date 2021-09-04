
import org.jml.linear_models.Perceptron;
import java.util.HashMap;

public class Test {
    public static void main(String[] args) {
        Perceptron model = new Perceptron();
        double[][][] x = {
                            {{1, 2, 3, 4},
                             {5, 6, 7, 4}},

                            {{1, 2, 3, 4},
                             {5, 6, 7, 4}},

                            {{1, 2, 3, 4},
                             {5, 6, 7, 4}},
                         };
        double[] y = {4, 5, 6};
        HashMap<String, Double> a = new HashMap<>();

        model.fit(x, y, a);
    }
}
