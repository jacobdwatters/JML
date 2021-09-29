import com.jml.core.Model;
import com.jml.core.ModelBucket;
import com.jml.linear_models.MultipleLinearRegression;
import com.jml.linear_models.PolynomialRegression;
import com.jml.util.ArrayUtils;

import java.util.HashMap;
import java.util.Map;

public class TempTests {
    public static void main(String[] args) {
        polynomialRegressionExample();
//        multipleRegressionExample();
    }


    static void polynomialRegressionExample() {
        double[] features = {1, 2, 3, 4, 5};
        double[] targets = {0, -7, -20, -39, -64};
        double[] tests = {-2, -1, 0};

        Model<double[], double[]> model = new PolynomialRegression();

        Map<String, Double> compileArgs = new HashMap<>();
        compileArgs.put("degree", 2.0);

        model.compile(compileArgs);
        ModelBucket results = model.fit(features, targets);

        double[] coefficients = results.getDoubleArr("coefficients");
        double[] predictions = model.predict(tests);

        System.out.println("\n\n" + ArrayUtils.asString(ArrayUtils.round(coefficients, 10)));
        System.out.println("Pred: " + ArrayUtils.asString(ArrayUtils.round(predictions, 10)) + "\n\n");
    }


    static void multipleRegressionExample() {
        double[][] features = { {2, 2},
                                {4, 5},
                                {6, 7}};

        double[] targets = {5, 6, 7};

        double[][] tests = {{9, 10},
                            {1, 8}};


        Model<double[][], double[]> model = new MultipleLinearRegression();

        model.compile();
        ModelBucket results = model.fit(features, targets);

        double[] coefficients = results.getDoubleArr("coefficients");
        double[] predictions = model.predict(tests);

        System.out.println("\n\n" + ArrayUtils.asString(ArrayUtils.round(coefficients, 10)));
        System.out.println("Pred: " + ArrayUtils.asString(ArrayUtils.round(predictions, 10)) + "\n\n");
    }
}
