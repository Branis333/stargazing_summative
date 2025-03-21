// lib/screens/home_screen.dart
import 'package:flutter/material.dart';
import 'package:geolocator/geolocator.dart';
import 'package:flutter_datetime_picker_plus/flutter_datetime_picker_plus.dart';
import 'package:intl/intl.dart';
import '../services/api_service.dart';
import '../models/prediction_result.dart';
import 'result_screen.dart';

class HomeScreen extends StatefulWidget {
  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final apiService = StargazingApiService();

  double? latitude;
  double? longitude;
  DateTime selectedDateTime = DateTime.now();
  bool isLoading = false;
  String locationName = "Select Location";

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Stargazing Predictor'),
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
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // App logo or image
              Image.asset('assets/Skywacth.png', height: 120),
              SizedBox(height: 24),

              // Location options
              Card(
                color: Colors.indigo.shade700,
                child: Column(
                  children: [
                    ListTile(
                      leading: Icon(Icons.my_location, color: Colors.white),
                      title: Text(
                        'Use Current Location',
                        style: TextStyle(color: Colors.white),
                      ),
                      trailing: Icon(
                        Icons.arrow_forward_ios,
                        color: Colors.white,
                      ),
                      onTap: () async {
                        await _getCurrentLocation();
                      },
                    ),
                    Divider(height: 1, color: Colors.indigo.shade500),
                    ListTile(
                      leading: Icon(
                        Icons.edit_location_alt,
                        color: Colors.white,
                      ),
                      title: Text(
                        'Enter Custom Location',
                        style: TextStyle(color: Colors.white),
                      ),
                      trailing: Icon(
                        Icons.arrow_forward_ios,
                        color: Colors.white,
                      ),
                      onTap: () {
                        _showLocationInputDialog();
                      },
                    ),
                  ],
                ),
              ),
              SizedBox(height: 8),

              // Selected location display
              if (latitude != null && longitude != null)
                Card(
                  color: Colors.indigo.shade600,
                  child: Padding(
                    padding: const EdgeInsets.all(12.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Selected Location:',
                          style: TextStyle(color: Colors.white70),
                        ),
                        SizedBox(height: 4),
                        Text(
                          locationName,
                          style: TextStyle(
                            color: Colors.white,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              SizedBox(height: 16),

              // Date & Time selector
              Card(
                color: Colors.indigo.shade700,
                child: ListTile(
                  leading: Icon(Icons.access_time, color: Colors.white),
                  title: Text(
                    DateFormat('MMM dd, yyyy - HH:mm').format(selectedDateTime),
                    style: TextStyle(color: Colors.white),
                  ),
                  trailing: Icon(Icons.arrow_forward_ios, color: Colors.white),
                  onTap: () {
                    DatePicker.showDateTimePicker(
                      context,
                      showTitleActions: true,
                      onConfirm: (date) {
                        setState(() {
                          selectedDateTime = date;
                        });
                      },
                      currentTime: selectedDateTime,
                    );
                  },
                ),
              ),

              Spacer(),

              // Predict button
              ElevatedButton(
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.amber,
                  foregroundColor: Colors.black,
                  padding: EdgeInsets.symmetric(vertical: 16),
                  textStyle: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                onPressed:
                    (latitude != null && longitude != null)
                        ? () async {
                          await _predictStargazing();
                        }
                        : null,
                child:
                    isLoading
                        ? CircularProgressIndicator(color: Colors.black)
                        : Text('PREDICT STARGAZING CONDITIONS'),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Future<void> _getCurrentLocation() async {
    setState(() {
      isLoading = true;
    });

    try {
      // Check location permissions
      LocationPermission permission = await Geolocator.checkPermission();
      if (permission == LocationPermission.denied) {
        permission = await Geolocator.requestPermission();
        if (permission == LocationPermission.denied) {
          throw Exception('Location permission denied');
        }
      }

      // Get current position
      Position position = await Geolocator.getCurrentPosition();
      setState(() {
        latitude = position.latitude;
        longitude = position.longitude;
        locationName =
            "Current Location (${position.latitude.toStringAsFixed(4)}, ${position.longitude.toStringAsFixed(4)})";
        isLoading = false;
      });
    } catch (e) {
      setState(() {
        isLoading = false;
      });
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Error getting location: $e')));
    }
  }

  Future<void> _predictStargazing() async {
    if (latitude == null || longitude == null) return;

    setState(() {
      isLoading = true;
    });

    try {
      final result = await apiService.predictStargazingQuality(
        latitude!,
        longitude!,
        selectedDateTime,
      );

      setState(() {
        isLoading = false;
      });

      // Navigate to results screen
      Navigator.push(
        context,
        MaterialPageRoute(
          builder:
              (context) => ResultScreen(
                result: PredictionResult.fromJson(result),
                latitude: latitude!,
                longitude: longitude!,
              ),
        ),
      );
    } catch (e) {
      setState(() {
        isLoading = false;
      });
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Error: $e')));
    }
  }

  void _showLocationInputDialog() {
    final latController = TextEditingController();
    final longController = TextEditingController();

    // Pre-fill with current values if available
    if (latitude != null) latController.text = latitude!.toString();
    if (longitude != null) longController.text = longitude!.toString();

    showDialog(
      context: context,
      builder:
          (context) => AlertDialog(
            backgroundColor: Colors.indigo.shade800,
            title: Text(
              'Enter Coordinates',
              style: TextStyle(color: Colors.white),
            ),
            content: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                TextField(
                  controller: latController,
                  keyboardType: TextInputType.numberWithOptions(
                    decimal: true,
                    signed: true,
                  ),
                  decoration: InputDecoration(
                    labelText: 'Latitude (-90 to 90)',
                    labelStyle: TextStyle(color: Colors.white70),
                    enabledBorder: UnderlineInputBorder(
                      borderSide: BorderSide(color: Colors.white54),
                    ),
                    focusedBorder: UnderlineInputBorder(
                      borderSide: BorderSide(color: Colors.amber),
                    ),
                  ),
                  style: TextStyle(color: Colors.white),
                ),
                SizedBox(height: 12),
                TextField(
                  controller: longController,
                  keyboardType: TextInputType.numberWithOptions(
                    decimal: true,
                    signed: true,
                  ),
                  decoration: InputDecoration(
                    labelText: 'Longitude (-180 to 180)',
                    labelStyle: TextStyle(color: Colors.white70),
                    enabledBorder: UnderlineInputBorder(
                      borderSide: BorderSide(color: Colors.white54),
                    ),
                    focusedBorder: UnderlineInputBorder(
                      borderSide: BorderSide(color: Colors.amber),
                    ),
                  ),
                  style: TextStyle(color: Colors.white),
                ),
              ],
            ),
            actions: [
              TextButton(
                child: Text('Cancel', style: TextStyle(color: Colors.white70)),
                onPressed: () => Navigator.pop(context),
              ),
              ElevatedButton(
                style: ElevatedButton.styleFrom(backgroundColor: Colors.amber),
                child: Text(
                  'Set Location',
                  style: TextStyle(color: Colors.black),
                ),
                onPressed: () {
                  // Validate and set the location
                  try {
                    final lat = double.parse(latController.text);
                    final lng = double.parse(longController.text);

                    if (lat < -90 || lat > 90) {
                      throw Exception('Latitude must be between -90 and 90');
                    }
                    if (lng < -180 || lng > 180) {
                      throw Exception('Longitude must be between -180 and 180');
                    }

                    setState(() {
                      latitude = lat;
                      longitude = lng;
                      locationName =
                          "Custom Location (${lat.toStringAsFixed(4)}, ${lng.toStringAsFixed(4)})";
                    });

                    Navigator.pop(context);
                  } catch (e) {
                    ScaffoldMessenger.of(context).showSnackBar(
                      SnackBar(content: Text('Invalid coordinates: $e')),
                    );
                  }
                },
              ),
            ],
          ),
    );
  }
}
