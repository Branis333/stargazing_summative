// lib/screens/result_screen.dart
import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import '../models/prediction_result.dart';

class ResultScreen extends StatelessWidget {
  final PredictionResult result;
  final double latitude;
  final double longitude;

  const ResultScreen({
    Key? key,
    required this.result,
    required this.latitude,
    required this.longitude,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Stargazing Prediction'),
        backgroundColor: Colors.indigo.shade800,
      ),
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Colors.indigo.shade900, Colors.black],
          ),
        ),
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Quality score card
              _buildQualityScoreCard(),
              SizedBox(height: 16),

              // Location info
              _buildInfoCard('Location', [
                'Reference: ${result.referenceLocation}',
                'Coordinates: (${latitude.toStringAsFixed(4)}, ${longitude.toStringAsFixed(4)})',
              ], icon: Icons.location_on),
              SizedBox(height: 16),

              // Time info
              _buildInfoCard('Time Information', [
                'Month: ${result.timeInfo.month}',
                'Day of year: ${result.timeInfo.dayOfYear}',
                'Hour: ${result.timeInfo.hour}:00',
                'Time category: ${result.timeInfo.timeCategory}',
                'Is night: ${result.timeInfo.isNight ? "Yes" : "No"}',
                'Is morning: ${result.timeInfo.isMorning ? "Yes" : "No"}',
              ], icon: Icons.access_time),
              SizedBox(height: 16),

              // Weather conditions chart
              _buildWeatherConditionsChart(),
              SizedBox(height: 24),

              // Message card
              Card(
                color: Colors.amber.shade700,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Row(
                    children: [
                      Icon(Icons.info_outline, color: Colors.white, size: 28),
                      SizedBox(width: 16),
                      Expanded(
                        child: Text(
                          result.message,
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildQualityScoreCard() {
    // Choose color based on quality
    Color qualityColor;
    String qualityText;

    if (result.stargazingQualityPercentage >= 80) {
      qualityColor = Colors.green;
      qualityText = "Excellent";
    } else if (result.stargazingQualityPercentage >= 60) {
      qualityColor = Colors.lightGreen;
      qualityText = "Good";
    } else if (result.stargazingQualityPercentage >= 40) {
      qualityColor = Colors.amber;
      qualityText = "Fair";
    } else {
      qualityColor = Colors.red;
      qualityText = "Poor";
    }

    return Card(
      color: Colors.indigo.shade700,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            Text(
              'Stargazing Quality',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
                color: Colors.white70,
              ),
            ),
            SizedBox(height: 12),
            Stack(
              alignment: Alignment.center,
              children: [
                SizedBox(
                  height: 150,
                  width: 150,
                  child: CircularProgressIndicator(
                    value: result.stargazingQualityPercentage / 100,
                    strokeWidth: 12,
                    backgroundColor: Colors.grey.shade800,
                    valueColor: AlwaysStoppedAnimation<Color>(qualityColor),
                  ),
                ),
                Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      '${result.stargazingQualityPercentage.toStringAsFixed(1)}%',
                      style: TextStyle(
                        fontSize: 36,
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                      ),
                    ),
                    Text(
                      qualityText,
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.w500,
                        color: qualityColor,
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildInfoCard(String title, List<String> details, {IconData? icon}) {
    return Card(
      color: Colors.indigo.shade800,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                if (icon != null) ...[
                  Icon(icon, color: Colors.white70),
                  SizedBox(width: 8),
                ],
                Text(
                  title,
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                  ),
                ),
              ],
            ),
            Divider(color: Colors.white24),
            ...details.map(
              (detail) => Padding(
                padding: const EdgeInsets.symmetric(vertical: 4.0),
                child: Text(
                  detail,
                  style: TextStyle(color: Colors.white70, fontSize: 14),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildWeatherConditionsChart() {
    // Prepare data for bar chart
    final conditions = result.predictedConditions;
    final data = [
      MapEntry('Cloud Cover', conditions.cloudCover),
      MapEntry('Humidity', conditions.humidity),
      MapEntry('PM2.5', conditions.airQualityPM25),
      MapEntry('PM10', conditions.airQualityPM10),
      MapEntry(
        'Visibility',
        conditions.visibilityKm * 10,
      ), // Scale up for visibility
    ];

    return Card(
      color: Colors.indigo.shade800,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Weather Conditions',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
                color: Colors.white,
              ),
            ),
            Divider(color: Colors.white24),
            SizedBox(height: 8),
            SizedBox(
              height: 200,
              child: BarChart(
                BarChartData(
                  alignment: BarChartAlignment.spaceAround,
                  maxY: 100,
                  barTouchData: BarTouchData(enabled: false),
                  titlesData: FlTitlesData(
                    show: true,
                    bottomTitles: AxisTitles(
                      sideTitles: SideTitles(
                        showTitles: true,
                        getTitlesWidget: (value, meta) {
                          // Custom titles for the X-axis
                          String text = '';
                          switch (value.toInt()) {
                            case 0:
                              text = 'Cloud';
                              break;
                            case 1:
                              text = 'Humidity';
                              break;
                            case 2:
                              text = 'PM2.5';
                              break;
                            case 3:
                              text = 'PM10';
                              break;
                            case 4:
                              text = 'Visibility';
                              break;
                          }
                          return Padding(
                            padding: const EdgeInsets.only(top: 8.0),
                            child: Text(
                              text,
                              style: TextStyle(
                                color: Colors.white60,
                                fontSize: 12,
                              ),
                            ),
                          );
                        },
                      ),
                    ),
                    leftTitles: AxisTitles(
                      sideTitles: SideTitles(
                        showTitles: true,
                        interval: 20,
                        getTitlesWidget: (value, meta) {
                          return Text(
                            value.toInt().toString(),
                            style: TextStyle(
                              color: Colors.white60,
                              fontSize: 12,
                            ),
                          );
                        },
                      ),
                    ),
                    topTitles: AxisTitles(
                      sideTitles: SideTitles(showTitles: false),
                    ),
                    rightTitles: AxisTitles(
                      sideTitles: SideTitles(showTitles: false),
                    ),
                  ),
                  gridData: FlGridData(show: true),
                  barGroups:
                      data
                          .asMap()
                          .map((index, item) {
                            // Color depends on the value and what's being measured
                            Color barColor;
                            if (index == 4) {
                              // Visibility - higher is better
                              barColor =
                                  item.value > 60
                                      ? Colors.green
                                      : item.value > 30
                                      ? Colors.amber
                                      : Colors.red;
                            } else {
                              // For other metrics, lower is better
                              barColor =
                                  item.value < 30
                                      ? Colors.green
                                      : item.value < 60
                                      ? Colors.amber
                                      : Colors.red;
                            }

                            return MapEntry(
                              index,
                              BarChartGroupData(
                                x: index,
                                barRods: [
                                  BarChartRodData(
                                    toY: item.value,
                                    color: barColor,
                                    width: 22,
                                    borderRadius: BorderRadius.only(
                                      topLeft: Radius.circular(6),
                                      topRight: Radius.circular(6),
                                    ),
                                  ),
                                ],
                              ),
                            );
                          })
                          .values
                          .toList(),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
