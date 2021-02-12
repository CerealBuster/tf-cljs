(ns cerealbuster.core
  (:require [cljsjs.tfjs]))

(defn tf-tensor [values & {:keys [shape dtype]}]
  (.tensor js/tf (clj->js values) shape dtype))

(defn tf-scalar [values & {:keys [dtype]}]
  (.scalar js/tf (clj->js values dtype)))

(defn tf-tensor1d [values & {:keys [dtype]}]
  (.tensor1d js/tf (clj->js values) dtype))

(defn tf-tensor2d [values & {:keys [shape dtype]}]
  (.tensor2d js/tf (clj->js values) shape dtype))

(defn tf-tensor3d [values & {:keys [shape dtype]}]
  (.tensor3d js/tf (clj->js values) shape dtype))

(defn tf-tensor4d [values & {:keys [shape dtype]}]
  (.tensor4d js/tf (clj->js values) shape dtype))

(defn tf-tensor5d [values & {:keys [shape dtype]}]
  (.tensor5d js/tf (clj->js values) shape dtype))

(defn tf-tensor6d [values & {:keys [shape dtype]}]
  (.tensor6d js/tf (clj->js values) shape dtype))

(defn tf-buffer [values & {:keys [shape dtype]}]
  (.buffer js/tf (clj->js values) shape dtype))

(defn tf-clone [x]
  (.clone js/tf (clj->js x)))

(defn tf-complex [real imag]
  (.complex js/tf (clj->js real) (clj->js imag)))

(defn tf-eye [num-rows &{:keys [num-columns batch-shape dtype]}]
  (.eye js/tf num-rows num-columns batch-shape dtype))

(defn tf-fill [shape value & {:keys [dtype]}]
  (.fill js/tf (clj->js shape) value dtype))

(defn tf-imag [i]
  (.imag js/tf (clj->js i)))

(defn tf-linspace [start stop num]
  (.linspace js/tf start stop num))

(defn tf-one-hot [indices depth &{:keys [on-value off-value] :or {on-value 1 off-value 0} }]
  (.oneHot js/tf (clj->js indices) depth on-value off-value))

(defn tf-ones [shape &{:keys [dtype]}]
  (.ones js/tf (clj->js shape) dtype))

(defn tf-ones-like [x]
  (.onesLike js/tf (clj->js x)))

(defn tf-print [x & {:keys [verbose]}]
  (.print js/tf (clj->js x) verbose))

(defn tf-range [start stop & {:keys [step dtype] :or {step 1}}]
  (if (and (< stop start ) (= step 1))
    (.range js/tf start stop -1 dtype)
    (.range js/tf start stop step dtype)))

(defn tf-real [i]
  (.real js/tf (clj->js i)))

(defn tf-truncated-normal [shape & {:keys [mean std-dev dtype seed]
                                    :or {mean 0 std-dev 1 dtype "float32"}}]
  (.truncatedNormal js/tf (clj->js shape) mean std-dev dtype seed))

(defn tf-variable [init-value &{:keys [trainable name dtype] :or {trainable true}}]
  (.variable js/tf init-value trainable name dtype))

(defn tf-zeros [shape & {:keys [dtype] :or [dtype "float32"]}]
  (.zeros js/tf (clj->js shape)))

(defn tf-zeros-like [shape]
  (.zerosLike js/tf (clj->js shape)))
