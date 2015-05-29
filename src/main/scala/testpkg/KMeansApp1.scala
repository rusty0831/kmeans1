package testpkg

import _root_.org.apache.spark.SparkConf
import _root_.org.apache.spark.SparkContext
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.clustering._
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.clustering._



/**
 * Created by rusty_lai on 2015/5/29.
 */
object KMeansApp1 {
  val conf = new SparkConf().setAppName("Kmeans Test 1")
  val sc = new SparkContext(conf)

  val rawData = sc.textFile("/root/kddcup")
  rawData.map(_.split(',').last).countByValue().toSeq.sortBy(_._2).reverse.foreach(println)

  val labelsAndData = rawData.map { line =>
    val buffer = line.split(',').toBuffer
    buffer.remove(1, 3)
    val label = buffer.remove(buffer.length-1)
    val vector = Vectors.dense(buffer.map(_.toDouble).toArray)
    (label,vector)
  }

  val parsedData = rawData.map { line =>
    val buffer = line.split(',').toBuffer
    buffer.remove(1, 3)
    val label = buffer.remove(buffer.length-1)
    val vector = Vectors.dense(buffer.map(_.toDouble).toArray)
    (vector)
  }

  val data = labelsAndData.values.cache()

  val kmeans = new KMeans()
  kmeans.setK(25)
  val model = kmeans.run(data)

  model.clusterCenters.foreach(println)

  val clusterLabelCount = labelsAndData.map { case (label,datum) =>
    val cluster = model.predict(datum)
    (cluster,label)
  }.countByValue

  clusterLabelCount.toSeq.sorted.foreach {
    case ((cluster,label),count) =>
      println(f"$cluster%1s$label%18s$count%8s")
  }

  println("Cost = " + model.computeCost(parsedData))
}
