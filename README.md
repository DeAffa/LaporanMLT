# Laporan Proyek Machine Learning - Muhammad Daffa Nurahman

## Domain Proyek
  Menurut data Kementerian Lingkungan Hidup dan Kehutanan Republik Indonesia, timbulan sampah domestik nasional mencapai sekitar 67,8 juta ton per tahun, tetapi hanya sekitar 11 % yang terkelola melalui daur ulang formal. Banyak Tempat Pembuangan Akhir (TPA) di kabupaten/kota beroperasi di atas 80 % kapasitas desain, menimbulkan risiko pencemaran dan masalah kesehatan masyarakat [1].

  Penelitian Putra et al. menunjukkan bahwa pemilahan manual sampah di tingkat rumah tangga di Jakarta memiliki tingkat kesalahan hingga 25 %, yang mengurangi efisiensi daur ulang dan meningkatkan kontaminasi limbah organik serta anorganik [2]. Hal ini menegaskan perlunya sistem klasifikasi otomatis berbasis machine learning untuk meningkatkan akurasi dan kecepatan sortir.

  Beberapa studi di Indonesia telah mengembangkan solusi ML/CV untuk klasifikasi sampah:
  
- Kadyanan et al. mengimplementasikan Convolutional Neural Network (CNN) untuk klasifikasi sampah padat—termasuk organik, plastik, kertas, dan logam—pada sampel lapangan di Bali, mencapai akurasi hingga 95,2 % tanpa memerlukan lisensi berbayar [3].

- Sari dan Prasetyo memanfaatkan MobileNetV2 dalam aplikasi mobile untuk membedakan sampah organik dan anorganik di Yogyakarta, dengan akurasi 90,2 % dan waktu inference rata rata 120 ms per citra [4].

- Tim Telkom University mengimplementasikan YOLOv5 pada CCTV untuk deteksi sampah plastik di daerah aliran sungai, memperoleh akurasi hingga 84 % dalam kondisi real time [5].

Berdasarkan data dan hasil studi tersebut, penelitian klasifikasi sampah berbasis machine learning di Indonesia penting untuk:
1.	Mengotomasi proses sortir dengan akurasi tinggi dan waktu inferensi rendah.
2.	Meningkatkan tingkat daur ulang nasional melalui pemilahan yang lebih akurat.
3.	Mengurangi beban operasional TPA dan dampak negatif lingkungan.

## Business Understanding
Untuk memastikan arah dan ruang lingkup proyek klasifikasi sampah berbasis _Machine Learning_ terdefinisi dengan jelas, berikut disajikan bagian Business Understanding sebagai landasan perencanaan dan evaluasi solusi :

1.  Problem Statements
  - Bagaimana mengklasifikasikan jenis sampah dari gambar secara otomatis dan akurat
  - Algoritma deep learning apa yang memberikan hasil terbaik untuk klasifikasi gambar sampah pada dataset yang tersedia

2.  Goals
   - Membangun sistem klasifikasi gambar sampah yang akurat dan efisien menggunakan deep learning
   - Membandingkan dua model deep learning populer dalam tugas klasifikasi gambar : EfficientNetB2 dan InceptionV3

3.  Solution Statements
Untuk mencapai tujuan tersebut, proyek ini akan mengimplementasikan dan membandingkan dua pendekatan :
-  EfficientNetB2 : Digunakan sebagai baseline model, EfficientNetB2 memiliki keunggulan dalam efisiensi komputasi dan akurasi pada klasifikasi gambar. Model ini akan digunakan dengan transfer learning dan ditambahkan beberapa layer dense dan dropout sebagai bentuk regulasi.
-  InceptionV3 : Sebagai model pembanding, InceptionV3 adalah arsitektur convolutional neural network yang dikenal dengan kemampuannya menangani kompleksitas visual yang tinggi melalui penggunaan modul inception. InceptionV3 juga akan diterapkan melalui transfer learning dengan struktur top model yang sebanding.

Kedua model akan dievaluasi dengan metrik **akurasi**, **loss**, dan metrik lain seperti **precision**, **recall**, serta **F1-score**. Perbandingan ini bertujuan untuk menentukan model mana yang lebih unggul dalam tugas klasifikasi gambar sampah ini.

# Data Understanding
<a href="https://www.kaggle.com/datasets/kaptenyasa/dataset-sampah">Dataset </a> yang digunakan dalam proyek ini merupakan kumpulan citra sampah yang berasal dari laman <a href="https://www.kaggle.com/"> Kaggle </a> yang telah terbagi ke dalam beberapa kategori. Dataset ini bisa diakses dengan tautan sebagai berikut: https://www.kaggle.com/datasets/kaptenyasa/dataset-sampah

Berdasarkan dataset tersebut, kita bisa melihat bahwa data citra yang ada pada dataset tersebut berjumlah 7000 gambar dengan rincian sebagai berikut :

- Organic : 1000 gambar
- Kertas : 1000 gambar
- Plastik : 1000 gambar
- Botol plastik : 1000 gambar
- Kardus : 1000 gambar
- Kaca : 1000 gambar
- Metal 1000 gambar

Berdasarkan eksplorasi awal terhadap dataset citra sampah, dilakukan pengambilan beberapa sampel visualisasi acak untuk data yang tersedia. Gambar-gambar yang ditampilkan dengan rasionya memberikan gambaran visual mengenai karakteristik dari data citra yang ada. Rasio gambar secara umum menunjukkan konsistensi proporsional, sehingga proses_Resize_ atau normalisasi ukuran pada tahap _preprocessing_ selanjutnya dapat dilakukan dengan efisien tanpa mengorbankan informasi penting dari gambar.

## Data Preparation
Pada tahap ini, dilakukan serangkaian proses untuk mempersiapkan data gambar sampah agar siap digunakan dalam proses pelatihan model klasifikasi berbasis deep learning. Proses data preparation dilakukan secara berurutan sebagai berikut :

1.  Memuat dan Mengorganisasi Dataset : Dataset yang digunakan merupakan kumpulan gambar sampah yang sudah dikategorikan ke dalam folder berdasarkan kelasnya.
2.  Resize Gambar : Ukuran asli gambar pada dataset tidak seragam. Oleh karena itu, seluruh gambar diubah ukurannya secara konsisten menjadi 224x224 piksel, agar sesuai dengan arsitektur input model _EfficientNetB1_ yang akan digunakan nantinya. Ini juga memastikan bahwa dimensi input antar gambar seragam selama proses pelatihan
3.  Rescaling (Normalisasi nilai piksel) : Seluruh gambar dinormalisasi dengan cara mengubah nilai piksel dari rentang 0-255 menjadi 0-1. Langkah ini penting untuk mempercepat proses pelatihan model dan mencegah dominasi nilai besar dalam proses komputasi jaringan saraf
4.  Augmentasi Data : Teknik augmentasi gambar diterapkan untuk memperkaya variasi data pelatihan dan mengurangi resiko overfitting./
5.  Splitting Data : Dataset dibagi menjadi tiga subset utama : Data pelatihan, data validasi, dan data testing. Proporsi pemisahan dilakukan sekitar 60:20:20. Hal ini bertujuan agar performa model dapat dievaluasi terhadap data yang belum pernah dilihat selama pelatihan, sehingga memberikan gambaran performa generalisasi model.

## Modelling
Dalam proyek ini digunakan dua arsitektur deep learning berbasis _Convolutional Neural Network (CNN)_, yaitu **EfficientNetB2** dan **InceptionV3**, keduanya dengan pendekatan transfer learning.

1.  EfficientNetB2
   -  Pretrained weights : ImageNet
   -  Layer Tambahan :
       -  GlobalAveragePooling2D
       -  Dropout (0.4, 0.3, 0.2)
       -  Dense (256 dan 128 unit, ReLU)
       -  Output : Dense dengan jumlah neuron sesuai jumlah kelas dan aktivasi softmax
  -  Optimizer : Adam
  -  Loss : Categorical Crossentropy
  -  Callback : EarlyStopping dan ModelCheckpoint
2.  InceptionV3
-  Pretrained weights : ImageNet
-  Include top : False
-  Input shape 224x224x3
-  Layer tambahan (sama dengan EfficientNetB2)
-  Optimizer : Adam
-  Loss : Categorical Crossentropy
-  Callback : EarlyStopping dan ModelCheckpoint

Masing-masing algoritma memilki kekurangan dan kelebihannya, yaitu :
1.  EfficientNetB2
   -  Kelebihan :
       -  Efisien secara komputasi dan memori
       -  Arsitektur yang ringan tetapi tetap mampu memberikan akurasi tinggi
       -  Cocok untuk pengujian cepat dan deployment
  -  Kekurangan :
      -  Bisa kurang akurat untuk dataset dengan kompleksitas tinggi jika tidak dilakukan fine-tuning secara mendalam
   
2.  InceptionV3
   -  Kelebihan :
       -  Mampu menangkap pola visual kompleks melalui struktur inception module
       -  Lebih mendalam dan lebih kuat dari banyak model CNN standar
    
  - Kekurangan :
      -  Lebih berat secara komputasi
      -  Cenderung membutuhkan lebih banyak data dan waktu pelatihan
   
## Evaluation
Dalam proyek klasifikasi gambar sampah ini, model dievaluasi menggunakan beberapa metrik yaitu **akurasi**, **precision**, **recall**, dan **F1-Score**. **Akurasi** digunakan untuk melihat seberapa banyak prediksi model yang benar yang benar dibandingkan dari seluruh data. **Precision** menunjukkan seberapa tepat model dalam memprediksi satu kelas tertentu, sedangkan **Recall** mengukur seberapa baik model dalam menemukan semua data dari kelas tersebut. **F1-Score** digunakan untuk menyeimbangkan precision dan recall agar hasil lebih adil. Metrik-metrik ini dipilih karena sesuai untuk tugas klasifikasi dengan banyak kelas, dan penting untuk mengetahui kinerja model tidak hanya keseluruhan, tetapi juga pada masing-masing kelas.
