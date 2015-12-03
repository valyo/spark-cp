package se.uu.farmbio.cp.alg

import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import se.uu.farmbio.cp.UnderlyingAlgorithm

//Define a SVMs UnderlyingAlgorithm
object SVM {
  def trainingProcedure(
      input: RDD[LabeledPoint], 
      numIterations: Int): (Vector => Double) = {
    //Training
//    val remappedInput = input.map(x => new LabeledPoint((x.label * 2) - 1, x.features))
    val model = SVMWithSGD.train(input,numIterations)
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
      score
    } else {
      -score
    }
  }
}