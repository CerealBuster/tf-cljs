(ns cerealbuster.layers
  (:require [cljsjs.tfjs]
            [cljs.core.async :refer [go chan <! >!]]
            [cljs.core.async.interop :refer-macros [<p!]]
            ))


(defn elu [& {:keys [alpha
                     input-shape
                     batch-input-shape
                     batch-size
                     dtype
                     name
                     trainable
                     weights
                     input-dtype]}]

  (.elu js/tf.layers (clj->js {:alpha alpha
                               :inputShape input-shape
                               :batchInputShape batch-input-shape
                               :batchSize batch-size
                               :dtype dtype
                               :name name
                               :trainable trainable
                               :weights weights
                               :inputDType input-dtype})))


(defn leaky-relu [& {:keys [alpha
                            input-shape
                            batch-input-shape
                            batch-size
                            dtype
                            name
                            trainable
                            weights
                            input-dtype]}]
  (.leakyReLU js/tf.layers (clj->js {:alpha alpha
                                     :inputShape input-shape
                                     :batchInputShape batch-input-shape
                                     :batchSize batch-size
                                     :dtype dtype
                                     :name name
                                     :trainable trainable
                                     :weights weights
                                     :inputDType input-dtype})))


(defn prelu [& {:keys [alpha-initializer
                       alpha-regularizer
                       alpha-constraint
                       shared-axes
                       input-shape
                       batch-input-shape
                       batch-size
                       dtype
                       name
                       trainable
                       weights
                       input-dtype]}]
  (.prelu js/tf.layers (clj->js {:alphaInitialier alpha-initializer
                                 :alphaRegularizer alpha-regularizer
                                 :alpha-constraint alpha-constraint
                                 :inputShape input-shape
                                 :batchInputShape batch-input-shape
                                 :batchSize batch-size
                                 :dtype dtype
                                 :name name
                                 :trainable trainable
                                 :weights weights
                                 :inputDType input-dtype})))


(defn relu [& {:keys [max-value
                     input-shape
                     batch-input-shape
                     batch-size
                     dtype
                     name
                     trainable
                     weights
                     input-dtype]}]

  (.reLU js/tf.layers (clj->js {:maxValue max-value
                               :inputShape input-shape
                               :batchInputShape batch-input-shape
                               :batchSize batch-size
                               :dtype dtype
                               :name name
                               :trainable trainable
                               :weights weights
                               :inputDType input-dtype})))


(defn softmax [& {:keys [axis
                         input-shape
                         batch-input-shape
                         batch-size
                         dtype
                         name
                         trainable
                         weights
                         input-dtype]}]

  (.softmax js/tf.layers (clj->js {:axis axis
                                   :inputShape input-shape
                                   :batchInputShape batch-input-shape
                                   :batchSize batch-size
                                   :dtype dtype
                                   :name name
                                   :trainable trainable
                                   :weights weights
                                   :inputDType input-dtype})))

(defn thresholded-relu [& {:keys [theta
                                  input-shape
                                  batch-input-shape
                                  batch-size
                                  dtype
                                  name
                                  trainable
                                  weights
                                  input-dtype]}]

  (.thresholdedReLU js/tf.layers (clj->js {:theta theta
                                           :inputShape input-shape
                                           :batchInputShape batch-input-shape
                                           :batchSize batch-size
                                           :dtype dtype
                                           :name name
                                           :trainable trainable
                                           :weights weights
                                           :inputDType input-dtype})))

(defn activation [& {:keys [activation
                            input-shape
                            batch-input-shape
                            batch-size
                            dtype
                            name
                            trainable
                            weights
                            input-dtype]}]

  (.activation js/tf.layers (clj->js {:activation activation
                                      :inputShape input-shape
                                      :batchInputShape batch-input-shape
                                      :batchSize batch-size
                                      :dtype dtype
                                      :name name
                                      :trainable trainable
                                      :weights weights
                                      :inputDType input-dtype})))

(defn dense [& {:keys [units
                       activation
                       use-bias
                       kernel-initializer
                       bias-initializer
                       input-dim
                       kernel-constraint
                       bias-constraint
                       kernel-reqularizer
                       bias-reqularizer
                       activity-regularizer
                       input-shape
                       batch-input-shape
                       batch-size
                       dtype
                       name
                       trainable
                       weights
                       input-dtype]}]

  (.dense js/tf.layers (clj->js {:units units
                                 :activation activation
                                 :useBias use-bias
                                 :kernelInitializer kernel-initializer
                                 :biasInitializer bias-initializer
                                 :inputDim input-dim
                                 :kernelConstraint kernel-constraint
                                 :biasConstraint bias-constraint
                                 :kernelRegularizer kernel-reqularizer
                                 :biasRegularizer bias-reqularizer
                                 :activityRegularizer activity-regularizer
                                 :inputShape input-shape
                                 :batchInputShape batch-input-shape
                                 :batchSize batch-size
                                 :dtype dtype
                                 :name name
                                 :trainable trainable
                                 :weights weights
                                 :inputDType input-dtype})))



(defn dropout [& {:keys [rate
                         noise-shape
                         seed
                         input-shape
                         batch-input-shape
                         batch-size
                         dtype
                         name
                         trainable
                         weights
                         input-dtype]}]

  (.dropout js/tf.layers (clj->js {:rate rate
                                   :noiseShape noise-shape
                                   :seed seed
                                   :inputShape input-shape
                                   :batchInputShape batch-input-shape
                                   :batchSize batch-size
                                   :dtype dtype
                                   :name name
                                   :trainable trainable
                                   :weights weights
                                   :inputDType input-dtype})))


(defn embedding [& {:keys [input-dim
                           output-dim
                           embeddings-initializer
                           embeddings-regularizer
                           activity-regularizer
                           embedding-constraint
                           mask-zero
                           input-shape
                           batch-input-shape
                           batch-size
                           dtype
                           name
                           trainable
                           weights
                           input-dtype]}]

  (.embedding js/tf.layers (clj->js {:inptuDim input-dim
                                     :outputDim output-dim
                                     :embeddingsInitializer embeddings-initializer
                                     :embeddingsRegularizer embeddings-regularizer
                                     :activityRegularizer activity-regularizer
                                     :embeddingConstraint embedding-constraint
                                     :maksZero mask-zero
                                     :inputShape input-shape
                                     :batchInputShape batch-input-shape
                                     :batchSize batch-size
                                     :dtype dtype
                                     :name name
                                     :trainable trainable
                                     :weights weights
                                     :inputDType input-dtype})))

(defn flatten [& {:keys [data-format
                         input-shape
                         batch-input-shape
                         batch-size
                         dtype
                         name
                         trainable
                         weights
                         input-dtype]}]

  (.flatten  js/tf.layers (clj->js { :dataFormat data-format
                                    :inputShape input-shape
                                    :batchInputShape batch-input-shape
                                    :batchSize batch-size
                                    :dtype dtype
                                    :name name
                                    :trainable trainable
                                    :weights weights
                                    :inputDType input-dtype})))


(defn permute [& {:keys [dims
                         input-shape
                         batch-input-shape
                         batch-size
                         dtype
                         name
                         trainable
                         weights
                         input-dtype]}]

  (.permute  js/tf.layers (clj->js { :dims dims
                                    :inputShape input-shape
                                    :batchInputShape batch-input-shape
                                    :batchSize batch-size
                                    :dtype dtype
                                    :name name
                                    :trainable trainable
                                    :weights weights
                                    :inputDType input-dtype})))

(defn reshape [& {:keys [target-shape
                         input-shape
                         batch-input-shape
                         batch-size
                         dtype
                         name
                         trainable
                         weights
                         input-dtype]}]

  (.reshape  js/tf.layers (clj->js { :targetShape target-shape
                                    :inputShape input-shape
                                    :batchInputShape batch-input-shape
                                    :batchSize batch-size
                                    :dtype dtype
                                    :name name
                                    :trainable trainable
                                    :weights weights
                                    :inputDType input-dtype})))


(defn spatial-dropout-1d [& {:keys [rate
                                    seed
                                    input-shape
                                    batch-input-shape
                                    batch-size
                                    dtype
                                    name
                                    trainable
                                    weights
                                    input-dtype]}]

  (.spatialDroput1d  js/tf.layers (clj->js {:rate rate
                                            :seed seed
                                            :inputShape input-shape
                                            :batchInputShape batch-input-shape
                                            :batchSize batch-size
                                            :dtype dtype
                                            :name name
                                            :trainable trainable
                                            :weights weights
                                            :inputDType input-dtype})))
(coment
 (def model (cerealbuster.models/tf-sequential))

 (cerealbuster.models/add-layer! model (dense :units 10 :input-shape [20] :activation "relu"))
 (cerealbuster.models/add-layer! model (dense :units 20 :activation "relu"))
 (cerealbuster.models/add-layer! model (dense :units 2 :activation "softmax"))

 (cerealbuster.models/summary model)

 (cerealbuster.models/compie-model! model :optimizer "adam" :metrics ["acc"] :loss "categoricalCrossentropy")

 (def x (cerealbuster.core/tf-ones [8 20]))
 (def y (cerealbuster.core/tf-ones [8 2]))

 (go (let [history (<p! (cerealbuster.models/fit model x y :batch-size 4 :epochs 3))]
       (print (js->clj (.-history history))))))
