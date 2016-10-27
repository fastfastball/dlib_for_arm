# dlib_for_arm

very fast face detection for ARM platform.

The code is based on dlib with the following enhancement
1) reduce work load : only use 1 filter(front looking) instead of 5 filters in frontal_face_detector.h
2) thread level parallelism : use 3 threads to do face detection 
3) SIMD : use arm neon to implement dlib/simd/

