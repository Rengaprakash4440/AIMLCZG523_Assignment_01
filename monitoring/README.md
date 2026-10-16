# Monitoring

The Heart Disease Prediction API is monitored using
Prometheus and Grafana running on a virtual machine.

## Architecture

Heart Disease API
       |
       | /metrics
       v
   Prometheus
       |
       v
     Grafana

## Prometheus

Prometheus collects metrics exposed by the prediction API.

Configuration:

`prometheus.yml`

## Grafana

Grafana uses Prometheus as the monitoring data source.

The exported dashboard is available at:

`heart-api-dashboard.json`

## Verification

1. Send prediction requests to the API.
2. Verify that the API appears as an UP target in Prometheus.
3. Verify application metrics using Prometheus.
4. Open Grafana and verify the Heart Disease API dashboard.

## Evidence

Screenshots demonstrate:
- Prometheus target status
- Prometheus metrics
- Grafana monitoring dashboard
