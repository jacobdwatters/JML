import com.jml.core.Model;
import com.jml.core.ModelBucket;
import com.jml.linear_models.MultipleLinearRegression;
import com.jml.linear_models.PolynomialRegression;
import com.jml.util.ArrayUtils;

import java.util.HashMap;
import java.util.Map;

public class TempTests {
    public static void main(String[] args) {
        multipleRegressionExample();
    }


    static void multipleRegressionExample() {
        double[][] features = { {1, 4, -3, 4},
                                {5, 6, -4, 8},
                                {9, -1, 11, 12},
                                {0, 1, 0.6, 1},
                                {5, 4, 3, 2}};

        double[] targets = {1, 9, -1, 6, 4};


        Model<double[][], double[]> model = new MultipleLinearRegression();

        model.compile();
        ModelBucket results = model.fit(features, targets);

        double[] coefficients = results.getDoubleArr("coefficients");

        System.out.println("\n\n" + model);
        System.out.println("\n\n" + ArrayUtils.asString(ArrayUtils.round(coefficients, 10)));

        Model m = Model.load("myModel234.mdl");

        System.out.println("\n\n" + m);
        System.out.println("\n\n" + ArrayUtils.asString(ArrayUtils.round(coefficients, 10)));
    }
}
