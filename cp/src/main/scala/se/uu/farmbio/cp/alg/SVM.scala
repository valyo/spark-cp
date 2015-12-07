package se.uu.farmbio.cp.alg

import org.apache.spark.mllib.classification.SVMModel
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.LBFGS
import org.apache.spark.mllib.optimization.LogisticGradient
import org.apache.spark.mllib.optimization.SquaredL2Updater
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD

import se.uu.farmbio.cp.UnderlyingAlgorithm

//Define a SVMs UnderlyingAlgorithm
object SVM {
  def trainingProcedure(
      input: RDD[LabeledPoint], 
      numIterations: Int): (Vector => Double) = {
    val numFeatures = input.take(1)(0).features.size
    val training = input.map(x => (x.label, MLUtils.appendBias(x.features))).cache()
    //Configuration
    val numCorrections = 10
    val convergenceTol = 1e-4
    val maxNumIterations = 20
    val regParam = 0.1
    val initialWeightsWithIntercept = Vectors.dense(new Array[Double](numFeatures + 1))

    val (weightsWithIntercept, loss) = LBFGS.runLBFGS(
      training,
      new LogisticGradient(),
      new SquaredL2Updater(),
      numCorrections,
      convergenceTol,
      maxNumIterations,
      regParam,
      initialWeightsWithIntercept)
    //Training
      val model = new SVMModel(
        Vectors.dense(weightsWithIntercept.toArray.slice(0, weightsWithIntercept.size - 1)),
        weightsWithIntercept(weightsWithIntercept.size - 1))
    model.predict
  }
}

class SVM(
  private val input: RDD[LabeledPoint],
  private val numIterations: Int)
  extends UnderlyingAlgorithm(
      SVM.trainingProcedure(input,numIterations)) {
  override def nonConformityMeasure(newSample: LabeledPoint) = {
    val score = predictor(newSample.features)
    if (newSample.label == 1.0) {
      -score
    } else {
      score
    }
  }
}
