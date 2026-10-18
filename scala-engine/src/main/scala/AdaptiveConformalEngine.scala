import scala.collection.mutable.ListBuffer

// ==========================================
// 1. DATA MODELS & CASE CLASSES
// ==========================================
case class ReviewRecord(id: String, text: String, aspect: String, rawSentiment: String, confidence: Double)
case class ConformalDecision(aspect: String, sentiment: String, confidence: Double, alpha: Double, status: String)
case class RiskPriorityRecord(reviewId: String, aspect: String, weightedRiskScore: Double, priorityTier: String)
case class SystemMetrics(totalProcessed: Int, acceptedCount: Int, abstainedCount: Int, empiricalErrorRate: Double)


// ==========================================
// 2. ALGORITHM 1: ADAPTIVE CONFORMAL ENGINE (ACI)
// ==========================================
class AdaptiveConformalEngine(targetErrorRate: Double = 0.10, initialAlpha: Double = 0.10, stepSize: Double = 0.05) {
  private var currentAlpha: Double = initialAlpha
  private val historyErrors = ListBuffer[Double]()

  def evaluateStreamItem(record: ReviewRecord, actualErrorOccurred: Boolean): ConformalDecision = {
    val dynamicThreshold = 1.0 - currentAlpha

    val decision = if (record.confidence < dynamicThreshold) {
      ConformalDecision(
        aspect = record.aspect,
        sentiment = "ABSTAIN",
        confidence = record.confidence,
        alpha = currentAlpha,
        status = s"ACI Abstention Triggered (Confidence ${record.confidence} < Threshold $dynamicThreshold)"
      )
    } else {
      ConformalDecision(
        aspect = record.aspect,
        sentiment = record.rawSentiment,
        confidence = record.confidence,
        alpha = currentAlpha,
        status = "Prediction Accepted within Adaptive Bounds"
      )
    }

    updateAlpha(actualErrorOccurred)
    decision
  }

  private def updateAlpha(errorOccurred: Boolean): Unit = {
    val errorIndicator = if (errorOccurred) 1.0 else 0.0
    historyErrors += errorIndicator
    currentAlpha = currentAlpha + stepSize * (targetErrorRate - errorIndicator)
    currentAlpha = math.max(0.01, math.min(0.50, currentAlpha))
  }
}


// ==========================================
// 3. ALGORITHM 2: WEIGHTED RISK PRIORITY SCORING (WRPSA)
// ==========================================
object WeightedRiskScoringAlgorithm {
  // Domain-specific criticality weights for electronic hardware aspects
  private val aspectCriticalityWeights: Map[String, Double] = Map(
    "battery life" -> 1.5,
    "display" -> 1.4,
    "audio" -> 1.2,
    "performance" -> 1.5,
    "default" -> 1.0
  )

  def computeRiskPriority(reviewId: String, aspect: String, confidence: Double, isAbstained: Boolean): RiskPriorityRecord = {
    val criticality = aspectCriticalityWeights.getOrElse(aspect.toLowerCase, aspectCriticalityWeights("default"))
    val uncertaintyGap = 1.0 - confidence
    val abstentionPenalty = if (isAbstained) 1.3 else 1.0
    
    val rawScore = criticality * uncertaintyGap * abstentionPenalty
    val normalizedScore = math.min(1.0, math.max(0.0, rawScore))

    val tier = normalizedScore match {
      case s if s >= 0.60 => "URGENT_HUMAN_AUDIT"
      case s if s >= 0.30 => "MODERATE_REVIEW"
      case _              => "LOW_RISK_AUTO_RESOLVED"
    }

    RiskPriorityRecord(reviewId, aspect, normalizedScore, tier)
  }
}


// ==========================================
// 4. METRICS & TELEMETRY TRACKER
// ==========================================
class ConformalMetricsTracker {
  private var total = 0
  private var accepted = 0
  private var abstained = 0
  private var errors = 0

  def recordDecision(decision: ConformalDecision, errorOccurred: Boolean): Unit = {
    total += 1
    if (decision.sentiment == "ABSTAIN") abstained += 1
    else {
      accepted += 1
      if (errorOccurred) errors += 1
    }
  }

  def generateReport(): SystemMetrics = {
    val errorRate = if (accepted > 0) errors.toDouble / accepted else 0.0
    SystemMetrics(total, accepted, abstained, errorRate)
  }
}


// ==========================================
// 5. MAIN EXECUTION PIPELINE (Python-Bridge Ready)
// ==========================================
object ACIExecutor {
  def main(args: Array[String]): Unit = {
    // Expecting arguments from Python: aspect, confidence, errorFlag
    // If no arguments are passed, it defaults to testing the sarcastic battery case
    val aspect = if (args.length > 0) args(0) else "battery life"
    val confidence = if (args.length > 1) args(1).toDouble else 0.82
    val errorFlag = if (args.length > 2) args(2).toBoolean else true

    val engine = new AdaptiveConformalEngine(targetErrorRate = 0.10)
    
    // Create record from input (Python or default fallback)
    val record = ReviewRecord("live-input", "Live test review", aspect, "POSITIVE", confidence)
    
    // Run Algorithm 1 (ACI) & Algorithm 2 (WRPSA)
    val conformalResult = engine.evaluateStreamItem(record, errorFlag)
    val isAbstained = conformalResult.sentiment == "ABSTAIN"
    val riskPriority = WeightedRiskScoringAlgorithm.computeRiskPriority(record.id, record.aspect, record.confidence, isAbstained)

    // Print out clean JSON format for Python subprocess parsing
    println(s"""{"aspect":"${conformalResult.aspect}","sentiment":"${conformalResult.sentiment}","confidence":${conformalResult.confidence},"status":"${conformalResult.status}","alpha":${conformalResult.alpha},"riskScore":${riskPriority.weightedRiskScore},"priorityTier":"${riskPriority.priorityTier}"}""")
  }
}