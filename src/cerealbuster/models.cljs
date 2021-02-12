(ns cerealbuster.models
  (:require [cljsjs.tfjs]
            [cljs.core.async :refer [go <! chan <!]]
            [cljs.core.async.interop :refer-macros [<p!]]))

(defn sequential [& {:keys [layers name]}]
  (.sequential js/tf (clj->js {:layers layers :name name})))

(defn tf-model [& {:keys [inputs outputs name]}]
  (.model js/tf (clj->js {:inputs inputs :outputs outputs :name name})))

(defn tf-input [& {:keys [shape batch name dtype sparse]}]
  (.input js/tf (clj->js {:shape shape :batch batch :name name :dtype dtype :sparse sparse})))

(defn tf-load-graph-model [model-url & {:keys [request-init
                                            on-progress
                                            fetch-func
                                            strict
                                            weight-path-prefix
                                            from-tf-hub
                                               weight-url-converter]}]
  (.loadGraphModel js/tf model-url (clj->js {:requestInit request-init
                                             :onProgress on-progress
                                             :fetchFunc fetch-func
                                             :strict strict
                                             :weightPathPrefix weight-path-prefix
                                             :fromTFHub from-tf-hub
                                             :weightUrlConverter weight-url-converter})))


(defn tf-load-layers-model [path-or-io-handler & {:keys [request-init
                                                         on-progress
                                                         fetch-func
                                                         strict
                                                         weight-path-prefix
                                                         from-tf-hub
                                                         weight-url-converter]}]

  (.loadLayersModel js/tf path-or-io-handler (clj->js {:requestInit request-init
                                                       :onProgress on-progress
                                                       :fetchFunc fetch-func
                                                       :strict strict
                                                       :weightPathPrefix weight-path-prefix
                                                       :fromTFHub from-tf-hub
                                                       :weightUrlConverter weight-url-converter})))


(defn tf-io-browser-download [& file-name-prefix]
  (.browserDownloads js/tf.io file-name-prefix))

(defn tf-io-broser-files [f]
  (.browserFiles js/tf.io f))

(defn tf-io-http [p & {:keys [weight-path-prefix fetch-func on-progress]}]
  (.http js/tf.io p (clj->js {:weightPathPrefix weight-path-prefix
                              :fetchFunc fetch-func
                              :onProgress on-progress})))

(defn tf-io-copy-model [source-url dest-url]
  (.copyModel js/tf.io source-url dest-url))

(defn tf-io-list-models []
  (.listModels js/tf.io))

(defn tf-io-move-model [source-url dest-url]
  (.moveModel js/tf.io source-url dest-url))

(defn tf-io-remove-model [url]
  (.removeModel js/tf.io url))

(defn tf-register-class [cls]
  (.registerClass js/tf cls))

(defn tf-functional []
  (new js/tf.Functional))

(defn tf-graph-model []
  (new js/tf.GraphModel))

(defn tf-layers-model []
  (new js/tf.LayersModel))

(defn tf-sequential []
  (new js/tf.Sequential))

(defn summary [model & {:keys [line-length positions print-fn] :or {print-fn #(js/console.log %)}}]
  (.summary model line-length positions print-fn))

(defn add-layer! [model layer]
  (.add model layer))

(defn compie-model! [model &{:keys [optimizer loss metrics]}]
  (.compile model (clj->js {:optimizer optimizer :loss loss :metrics metrics})))

(defn evaluate-model [model x y & {:keys [batch-size verbose sample-weight steps]}]
  (.evaluate model x y (clj->js {:batchSize batch-size
                                 :verbose verbose
                                 :sampleWeight sample-weight
                                 :steps steps})))

(defn evaluate-dataset [model dataset & {:keys [batches verbose]}]
  (.evaluateDataset model (clj->js {:batches batches :verbose verbose})))

(defn predict [model x & {:keys [batch-size verbose]}]
  (.predict model x (clj->js {:batchSize batch-size :verbose verbose})))

(defn predict-on-batch [model x]
  (.predictOnBatch model x))

(defn fit [model x y & {:keys [batch-size
                                     epochs
                                     verbose
                                     callbacks
                                     validation-split
                                     validation-data
                                     shuffle
                                     class-weight
                                     sample-weight
                                     initial-epoch
                                     steps-per-epoch
                                     validation-steps
                                     yield-every]}]

  (.fit model x y (clj->js {:batchSize batch-size
                            :epochs epochs
                            :verbose verbose
                            :callbacks callbacks
                            :validationSplit validation-split
                            :shuffle shuffle
                            :classWeight class-weight
                            :sampleWeight sample-weight
                            :initialEpoch initial-epoch
                            :stepsPerEpoch steps-per-epoch
                            :validtionSteps validation-steps
                            :yeldEvery yield-every})))

(defn fit-dataset [model x y & {:keys [batch-size
                                     epochs
                                     verbose
                                     callbacks
                                     validation-split
                                     validation-data
                                     shuffle
                                     class-weight
                                     sample-weight
                                     initial-epoch
                                     steps-per-epoch
                                     validation-steps
                                     yield-every]}]

  (.fitDataset model x y (clj->js {:batchSize batch-size
                            :epochs epochs
                            :verbose verbose
                            :callbacks callbacks
                            :validationSplit validation-split
                            :shuffle shuffle
                            :classWeight class-weight
                            :sampleWeight sample-weight
                            :initialEpoch initial-epoch
                            :stepsPerEpoch steps-per-epoch
                            :validtionSteps validation-steps
                            :yeldEvery yield-every})))

(defn train-on-batch [model x y]
  (.trainOnBatch x y))

(defn save [model handler-or-url & {:keys [trainable-only include-optimizer]
                                    :or {include-optimzer false}}]
  (.save model handler-or-url (clj->js {:trainableOnly trainable-only
                                        :includeOptimizer include-optimizer})))


(defn get-layer [model & name index]
  (.getLayer model))


(defn tf-symbolic-tensor []
  js/tf.SymbolicTensor)


(defn tf-deregister-op [name]
  (.deregisterOp js/tf name))

(defn tf-get-registered-op [name]
  (.getRegisteredOp js/tf name))

(defn tf-register-op [name op-fn]
  (.registerOp js/tf name op-fn))



(comment
  (def mobilenet (atom nil))

  (go
    (let [model (<p! (tf-load-graph-model "https://storage.googleapis.com/tfjs-models/savedmodel/mobilenet_v2_1.0_224/model.json"))]

      (reset! mobilenet model)))

  (def zeros (cerealbuster.core/tf-zeros [1 224 224 3]))
  (.predict @mobilenet zeros))

(def grm (tf-graph-model))
(def model (tf-sequential))

(.add model (js/tf.layers.dense (clj->js {:units 1 :inputShape [1]})))
