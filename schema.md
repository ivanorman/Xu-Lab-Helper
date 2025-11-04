# Experiment Metadata JSON Schema Report

This report contains the JSON schema derived from the brainstorm notes for defining experiment metadata in a standardized way.

## JSON Schema

The following schema outlines the structure and data types for capturing experiment metadata. It includes sections for optics parameters, sample information, and general experiment details.

```JSON
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Experiment Metadata",
  "description": "Schema for defining experiment metadata",
  "type": "object",
  "properties": {
    "optics_parameters": {
      "type": "object",
      "description": "Parameters related to the optical setup",
      "properties": {
        "exposure_time": {
          "type": "number",
          "description": "Exposure time in seconds"
        },
        "laser_power": {
          "type": "number",
          "description": "Laser power in milliwatts (mW) or other relevant unit"
        },
        "frames_per_exposure": {
          "type": "integer",
          "description": "Number of frames captured per exposure"
        },
        "center_lambda": {
          "type": "number",
          "description": "Center wavelength in nanometers (nm)"
        },
        "temperature": {
          "type": "number",
          "description": "Temperature in Kelvin (K) or Celsius (C)"
        },
        "magnetic_field": {
          "type": "number",
          "description": "Magnetic field strength in Tesla (T)"
        },
        "BN_thickness": {
          "type": "number",
          "description": "Boron Nitride thickness in nanometers (nm), if applicable"
        },
        "grating": {
          "type": "string",
          "description": "Description or model of the grating used"
        },
        "spot_on_the_sample": {
          "type": "string",
          "description": "Description of the laser spot on the sample (e.g., size, location)"
        },
        "scanner_voltage": {
          "type": "number",
          "description": "Scanner voltage in Volts (V)"
        },
        "excitation_wavelengths": {
          "type": "array",
          "items": {
            "type": "number"
          },
          "description": "Excitation wavelength(s) in nanometers (nm)"
        },
        "excitation_collection_polarization": {
          "type": "string",
          "description": "Details about excitation and collection polarization"
        },
        "rotation_mount_theta": {
          "type": "number",
          "description": "Rotation mount angle theta in degrees"
        }
      }
    },
    "sample_information": {
      "type": "object",
      "description": "Details about the sample used in the experiment",
      "properties": {
        "sample_name_or_type": {
          "type": "string",
          "description": "Name or type of the sample (e.g., material, specific identifier)"
        },
        "material": {
          "type": "string",
          "description": "Primary material of the sample"
        },
        "angle": {
          "type": "number",
          "description": "Angle of the sample or a feature, if applicable (in degrees)"
        },
        "layer_number": {
          "type": "integer",
          "description": "Layer number if the sample is layered"
        },
        "natural_language_description": {
          "type": "string",
          "description": "A free-text description of the sample"
        }
      },
      "required": [
        "sample_name_or_type"
      ]
    },
    "experiment_details": {
      "type": "object",
      "description": "General details about the experiment",
      "properties": {
        "experiment_type": {
          "type": "string",
          "enum": ["Integrated PL", "RMCD", "Resonance", "other"],
          "description": "Type of experiment conducted"
        },
        "custom_experiment_type_description": {
            "type": "string",
            "description": "Description if experiment_type is '''other'''"
        },
        "interpretation_notes": {
          "type": "string",
          "description": "Lines in interpretation for what was observed"
        },
        "post_assigned_effects_interpretation": {
          "type": "string",
          "description": "Post-assigned interpretation of observed effects"
        },
        "post_measurement_data_files": {
          "type": "array",
          "items": {
            "type": "string",
            "format": "uri-reference"
          },
          "description": "List of URIs or paths to post-measurement data files"
        },
        "latex_formatted_summary": {
            "type": "string",
            "description": "Natural language summary formatted in LaTeX, if available"
        }
      },
      "required": [
        "experiment_type"
      ]
    }
  },
  "required": [
    "sample_information",
    "experiment_details"
  ]
}
```

## Key Considerations:

* **Experiment Type**: The `experiment_type` field includes an "other" option. If selected, the `custom_experiment_type_description` field should be used for further details.
* **Descriptive Fields**: Textual descriptions are supported for sample details (`natural_language_description`), interpretation notes, and post-assigned effects.
* **Data Files**: The `post_measurement_data_files` field is an array intended to store URIs or paths to related data files.
* **LaTeX Formatting**: A `latex_formatted_summary` field is available for a LaTeX-formatted summary of the experiment, if applicable.
* **Required Fields**: `sample_name_or_type` (within `sample_information`) and `experiment_type` (within `experiment_details`) are mandatory. The `sample_information` and `experiment_details` objects themselves are also required at the top level.

