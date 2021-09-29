package com.jml;

import com.jml.core.Model;
import com.jml.core.ModelBucket;
import com.jml.linear_models.PolynomialRegression;
import com.jml.util.ArrayUtils;

import java.util.HashMap;

public class TempTest {

    public static void main(String[] args) {
        double[] x = new double[]{1.49, 3.03, 0.57, 5.74, 3.51, 3.73, 2.98, -0.18, 6.23, 3.38, 2.15, 2.1, 3.93, 2.47, -0.41};
        double[] y = new double[]{44.6, 57.8, 49.9, 61.3, 49.6, 61.8, 49.0, 44.7, 59.2, 53.9, 46.5, 54.7, 50.3, 51.2, 45.7};
        double[] tests = new double[]{-1.2, 0, 1, 2, 3.14, 4};

        Model<double[], double[]> model = new PolynomialRegression();

        HashMap<String, Double> arguments = new HashMap<>();
        arguments.put("degree", 6.0);

        model.compile(arguments);
        ModelBucket fitResults = model.fit(x, y);
        double[] values = ArrayUtils.round(fitResults.getDoubleArr("coefficients"), 8);
        double[] predictions = ArrayUtils.round(model.predict(tests), 7);

        System.out.println("\n\n" + ArrayUtils.asString(values) + "\n\n");
        System.out.println("\n\n" + ArrayUtils.asString(predictions) + "\n\n");
    }
}
