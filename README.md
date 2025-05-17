# Laporan Proyek Machine Learning - Muhammad Daffa Nurahman

## Domain Proyek
Industri otomotif saat ini terus berkembang dengan cepat, sehingga banyak pilihan kendaraan yang berbeda-beda dari segi harga, kapasitas, dan fitur keamanan. Hal ini membuat konsumen dan produsen perlu cara yang cepat dan tepat untuk menilai kualitas atau kelayakan kendaraan. Penilaian secara manual memakan waktu dan bisa bersifat subjektif. Oleh karena itu, diperlukan solusi otomatis menggunakan machine learning yang dapat mengklasifikasikan kendaraan berdasarkan spesifikasi teknisnya.

Dataset Car Evaluation yang dibuat oleh Bohanec dan Rajkovic (1998) banyak digunakan dalam penelitian klasifikasi kendaraan. Dataset ini memuat fitur-fitur seperti harga beli, biaya perawatan, jumlah pintu, kapasitas penumpang, ukuran bagasi, dan tingkat keamanan, yang kemudian diklasifikasikan menjadi kategori kelayakan kendaraan : **unacc**, **acc**, **good**, dan **vgood**. Dengan menggunakan algoritma klasifikasi, model machine learning dapat membantu memprediksi kategori kendaraan secara cepat dan akurat, yang bermanfaat bagi konsumen dan pelaku industri otomotif.

## Business Understanding
Dalam dunia otomotif, konsumen dan produsen harus bisa menilai apakah suatu kendaraan layak digunakan atau tidak, berdasarkan parameter seperti harga, kapasitas, dan keamanan. Namun, proses evaluasi manual membutuhkan waktu dan bisa bersifat subjektif karena tergantung pada opini masing-masing individu.

1.  Problem Statements

Berikut adalah permasalahan utama yang ingin diselesaikan pada proyek ini : 
  -  Bagaimana cara mengklasifikasikan kendaraan ke dalam kelayakan (unacceptable, acceptable, good, very good) secara otomatis dan akurat
  -  Bagaimana membangun model klasifikasi yang mampu bekerja dengan fitur **kategorikal**, serta bisa diinterpretasikan dan ditingkatkan performanya

2.  Goals

Adapun tujuan utama dari proyek ini adalah :
  -  Mengembangkan model machine learning untuk **mengklasifikasikan kelayakan kendaraan** berdasarkan fitur spesifikasi seperti harga, jumlah, pintu, dan tingkat keamanan.
  -  Membandingkan beberapa model klasifikasi untuk **menemukan model terbaik** dengan akurasi tinggi dan kesalahan klasifikasi yang rendah.
  -  Menggunakan metrik evaluasi seperti **akurasi**, **confusion matrix**, dan **F1-Score** untuk mengukur performa model secara objektif.
3.  Solution Statement

Untuk mencapai tujuan klasifikasi kelayakan kendaraan secara otomatis, proyek ini mengusulkan beberapa pendekatan berbasis machine learning :
  -  **Decision Tree Classifier** sebagai baseline : Algoritma ini dipilih karena sifatnya yang mudah dipahami, cepat dan sangat cocok untuk data dengan fitur kategorikal seperti pada dataset kendaraan ini. Decision Tree juga memberikan struktur aturan yang jelas dalam pengambilan keputusan, sehingga sangat bermanfaat untuk interpretasi awal terhadap data.
  -  **Random Forest Classifier** sebagai pembanding : Model ini merupakan pengembangan dari decision tree yang terdiri dari kumpulan pohon keputusan (ensemble), sehingga dapat meningkatkan akurasi sekaligsu mengurangi resiko overfitting. Random Forest dinilai mampu memberikan performa yang lebih stabil dan akuran pada data tabular seperti dataset yang digunakan dalam proyek ini.

Kinerja semua model akan diukur menggunakan beberapa metrik evaluasi, yaitu **akurasi** untuk mengetahui proporsi prediksi yang benar secara keseluruhan, **confusion matrix** untuk melihat distribusi kesalahan antar kelas, serta **F1-Score** untuk menangani kemunkinan ketidakseimbangan antar kategori kelayakan kendaraan. Dengan pendekatan ini, diharapkan dapat diperoleh model klasifikasi kendaraan yang tidak hanya akurat, tetapi juga efisien dan dapat digunakan untuk mendukung pengambilan keputusan secara otomatis.

# Data Understanding
Proyek ini menggunakan dataset bernama <a href="https://archive.ics.uci.edu/dataset/19/car+evaluation"> Car Evaluation Dataset </a> yang berasal dari laman <a href="https://archive.ics.uci.edu/"> UCI Machine Learning Repository </a> yang telah terbagi ke dalam beberapa kategori. Dataset ini bisa diakses dengan tautan sebagai berikut: https://archive.ics.uci.edu/dataset/19/car+evaluation

Dataset ini terdiri dari **1.728 baris data** dan **6 fitur** (atribut) input serta **1 label** (kelas) output. Semua fitur pada dataset ini bersifat **kategorikal**, dan tidak terdapat nilai kosong atau duplikat, sehingga data berada dalam kondisi bersih dan siap digunakan untuk analisis dan pelatihan model.

Berdasarkan eksplorasi awal terhadap dataset citra sampah, dilakukan pengambilan beberapa sampel visualisasi acak untuk data yang tersedia. Gambar-gambar yang ditampilkan dengan rasionya memberikan gambaran visual mengenai karakteristik dari data citra yang ada. Rasio gambar secara umum menunjukkan konsistensi proporsional, sehingga proses_Resize_ atau normalisasi ukuran pada tahap _preprocessing_ selanjutnya dapat dilakukan dengan efisien tanpa mengorbankan informasi penting dari gambar.

## Data Preparation
Pada tahap ini, dilakukan serangkaian proses untuk mempersiapkan data gambar sampah agar siap digunakan dalam proses pelatihan model klasifikasi berbasis deep learning. Proses data preparation dilakukan secara berurutan sebagai berikut :

1.  Memuat dan Mengorganisasi Dataset : Dataset yang digunakan merupakan kumpulan gambar sampah yang sudah dikategorikan ke dalam folder berdasarkan kelasnya.
2.  Resize Gambar : Ukuran asli gambar pada dataset tidak seragam. Oleh karena itu, seluruh gambar diubah ukurannya secara konsisten menjadi 224x224 piksel, agar sesuai dengan arsitektur input model _EfficientNetB1_ yang akan digunakan nantinya. Ini juga memastikan bahwa dimensi input antar gambar seragam selama proses pelatihan
3.  Rescaling (Normalisasi nilai piksel) : Seluruh gambar dinormalisasi dengan cara mengubah nilai piksel dari rentang 0-255 menjadi 0-1. Langkah ini penting untuk mempercepat proses pelatihan model dan mencegah dominasi nilai besar dalam proses komputasi jaringan saraf
4.  Augmentasi Data : Teknik augmentasi gambar diterapkan untuk memperkaya variasi data pelatihan dan mengurangi resiko overfitting./
5.  Splitting Data : Dataset dibagi menjadi tiga subset utama : Data pelatihan, data validasi, dan data testing. Proporsi pemisahan dilakukan sekitar 60:20:20. Hal ini bertujuan agar performa model dapat dievaluasi terhadap data yang belum pernah dilihat selama pelatihan, sehingga memberikan gambaran performa generalisasi model.

## Modelling
Proses pemodelan dilakukan melalui beberapa langkah sebagai berikut : 
1.  Pre-trained Model : Menggunakan **EfficientNetB3** dengan bobot pralatih dari ImageNet dan tanpa bagian fully connected layer pada bagian atas (top layer).
2.  Layer tambahan : Ditambahkan beberapa layer seperti GlobalAveragePooling2D, beberapa Dense layer, dan dropout untuk mengurangi overfitting
3.  Loss Function : **categorical_crossentropy** karena data diklasifikasikan dalam beberapa kelas.
4.  Optimizer : **Adam**, dengan learning rate yang disesuaikan melalui tuning
5.  Metrics : Akurasi digunakan sebagai metrik utama untuk evaluasi performa

Kelebihan dan kekurangan EfficientNetB3
-  Kelebihan :
    -  Efisien secara komputasi : lebih kecil dan lebih cepat dibanding banyak model lain dengan akurasi setara
    -  Performa tinggi pada berbagai tugas klasifikasi gambar, termasuk dataset yang kompleks
    -  Mudah diadaptasi melalui _fine-tuning_ dan kompatibel dengan _transfer learning_
-  Kekurangan :
    -  Memerlukan input gambar dengan resolusi tertentu (ukuran input lebih besar dari model ringan)
    -  Latensi bisa meningkat pada perangkat keras dengan memori rendah karena ukuran model menengah
    -  Tidak selalu optimal tanpa penyesuaian hyperparameter untuk kasus spesifik  
   
## Evaluation
Pada tahap evaluai, dilakukan pengukuran kinerja model menggunakan metrik yang sesuai dengan konteks klasifikasi multi-kelas untuk gambar sampah. Untuk mengevaluasi kinerja model, digunakan dua metrik utama, yaitu :
1.  Accuracy (akurasi)
   Akurasi merupakan metrik yang digunakan untuk mengukur seberapa banyak prediksi model yang benar dibandingkan dengan seluruh prediksi yang dilakukan.
2.  Loss (Categorical Crossentropy Loss)
   Loss function yang digunakan adalah _categorical crossentropy_, yang mengukur seberapa jauh prediksi model dari nilai sebenarnya dalam bentuk probabilitas.

Berdasarkan pelatihan dan validasi model menggunakan _EfficientNetB3_, model menunjukkan performa sebagai berikut : 
-  **Akurasi pada data validasi** mencapai sekitar **0.9014** atau setara dengan **90%**
-  **Loss pada data validasi** menunjukkan nilai yang relatif rendah, yaitu sebesar **0.3180** atau setara dengan **30%**

Penggunaan metrik akurasi dan loss telah memberikan gambaran yang cukup jelas terhadap performa model dalam mengklasifikasikan gambar sampah. Akurasi yang cukup tinggi menunjukkan bahwa model mampu mengenali pola dengan baik, sementara nilai loss yang rendah menandakan bahwa prediksi model cenderung dekat dengan label sebenarnya.
