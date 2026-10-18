add_ut_test(qa_CrossSpectralMatrix)
target_link_libraries(qa_CrossSpectralMatrix PRIVATE gr-beamforming gr-testing)
add_ut_test(qa_Beamformer)
target_link_libraries(qa_Beamformer PRIVATE gr-beamforming gr-testing)
add_ut_test(qa_ArrayScene)
target_link_libraries(
  qa_ArrayScene
  PRIVATE gr-beamforming
          gr-localisation
          gr-basic
          gr-fourier
          gr-math
          gr-testing)
