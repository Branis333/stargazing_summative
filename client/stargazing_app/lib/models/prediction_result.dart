class PredictionResult {
  final double stargazingQualityPercentage;
  final String referenceLocation;
  final WeatherConditions predictedConditions;
  final bool isNight;
  final TimeInfo timeInfo;
  final String message;

  PredictionResult({
    required this.stargazingQualityPercentage,
    required this.referenceLocation,
    required this.predictedConditions,
    required this.isNight,
    required this.timeInfo,
    required this.message,
  });

  factory PredictionResult.fromJson(Map<String, dynamic> json) {
    return PredictionResult(
      stargazingQualityPercentage: json['stargazing_quality_percentage'],
      referenceLocation: json['reference_location'],
      predictedConditions: WeatherConditions.fromJson(json['predicted_conditions']),
      isNight: json['is_night'],
      timeInfo: TimeInfo.fromJson(json['time_info']),
      message: json['message'],
    );
  }
}

class WeatherConditions {
  final double cloudCover;
  final double humidity;
  final double airQualityPM25;
  final double airQualityPM10;
  final double visibilityKm;

  WeatherConditions({
    required this.cloudCover,
    required this.humidity,
    required this.airQualityPM25,
    required this.airQualityPM10,
    required this.visibilityKm,
  });

  factory WeatherConditions.fromJson(Map<String, dynamic> json) {
    return WeatherConditions(
      cloudCover: json['cloud_cover'],
      humidity: json['humidity'],
      airQualityPM25: json['air_quality_PM2.5'],
      airQualityPM10: json['air_quality_PM10'],
      visibilityKm: json['visibility_km'],
    );
  }
}

class TimeInfo {
  final int month;
  final int dayOfYear;
  final int hour;
  final bool isNight;

  TimeInfo({
    required this.month,
    required this.dayOfYear,
    required this.hour,
    required this.isNight,
  });

  factory TimeInfo.fromJson(Map<String, dynamic> json) {
    return TimeInfo(
      month: json['month'],
      dayOfYear: json['day_of_year'],
      hour: json['hour'],
      isNight: json['is_night'],
    );
  }
}