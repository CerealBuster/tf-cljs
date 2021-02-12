(ns cerealbuster.data
  (:require [cljsjs.tfjs]
            [cljs.core.async :refer [go <!]]
            [cljs.core.async.interop :refer-macros [<p!]]))

(defn tf-data-array [items]
  (js/tf.data.array (clj->js items)))

(defn tf-data-csv [src & csv-config]
  (js/tf.data.csv src (clj->js csv-config)))

(defn tf-data-generator [generator]
  (js/tf.data.generator generator))

(defn tf-data-microphone [&{:keys [fftSize
                                   columnTruncateLength
                                   numFramesPerSpectogram
                                   sampleRateHz
                                   includeWaveform]}]
  (let [config {:fftSize fftSize
                :columnTruncateLength columnTruncateLength
                :numFramesPerSpectogram numFramesPerSpectogram
                :sampleRateHz sampleRateHz
                :includeWaveform includeWaveform}]

    (.microphone js/tf.data (clj->js config))))


(defn tf-data-webcam [webcamVideoElement &{:keys [facingMode
                                                  deviceId
                                                  resizeWidth
                                                  resizeHeight
                                                  centerCrop]}]
  (let [config {:facingMode facingMode
                :deviceId deviceId
                :resizeWidth resizeWidth
                :resizeHeight resizeHeight
                :centerCrop centerCrop}]

    (.webcam js/tf.data webcamVideoElement (clj->js config))))



(defn tf-data-zip [dataset]
  (.zip js/tf.data (clj->js dataset)))

(defn tf-data-csv-dataset []
  (js/tf.data.CSVDataset))


(defn tf-data-dataset []
  (js/tf.data.Dataset))


(comment
  (def ds1 (tf-data-array [{:a 1} {:a 2} {:a 3}]))
  (def ds2 (tf-data-array [{:b 4} {:b 5} {:b 6}]))
  (def ds3 (tf-data-zip [ds1 ds2]))
  (js/console.log ds1)

  (go  (print (js->clj (<p! (.toArray ds3))))))

(comment
  (go (let [mic (<p! (tf-data-microphone {:fftSize 1024
                                          :columnTruncateLength 232
                                          :numFramesPerSpectogram 43
                                          :sampleRateHz 44100
                                          :includeSpectogram true
                                          :includeWaveform true}))
            audiodata (<p! (.capture mic))
            spectrogram (.-spectrogram audiodata)
            ]
        (print spectrogram)
        (.stop mic))))

(comment
  (go (let [videoEl (js/document.createElement "video")
            cam (<p! (tf-data-webcam videoEl))
            img (<p! (.capture cam))]
        (print img)
        (.stop cam))))
