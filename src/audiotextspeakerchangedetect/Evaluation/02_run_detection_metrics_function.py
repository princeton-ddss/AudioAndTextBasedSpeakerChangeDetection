from speechmlpipeline.Evaluation.SpeakerChangeDetection.metrics_main_function import calculate_detection_metrics

prediction_output_path = '/Users/jf3375/Desktop/modern_family/evaluation/predictions'
labelled_data_csv_path = '/Users/jf3375/Desktop/evaluation_data/VoxConverse/test_csv'
csv_filename = 'aepyx.csv'
evaluation_df = calculate_detection_metrics(prediction_output_path, labelled_data_csv_path, csv_filename, tolerance=0.5)
