use hdk::prelude::*;

// Constants for physical laws
const SPEED_OF_SOUND: f64 = 343.0; // meters / second
const CLOCK_DRIFT_BUFFER: f64 = 0.050; // 50 milliseconds

/// An individual sensor's reading
#[derive(Clone, Debug, Serialize, Deserialize, SerializedBytes)]
pub struct SensorObservation {
    pub sensor_id: String,
    pub x_coord: f64,
    pub y_coord: f64,
    pub t_arrival: f64, // System epoch time in seconds
}

/// The bundled claim containing all observations for a single gunshot.
/// This structure makes the validation 100% deterministic.
#[hdk_entry_helper]
#[derive(Clone, Debug)]
pub struct AcousticTriangulationClaim {
    pub observations: Vec<SensorObservation>, // Must contain at least 3 distinct observations
}

// Modern entry definition registration
#[hdk_entry_defs]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    AcousticTriangulationClaim(AcousticTriangulationClaim),
}

/// Exposes an endpoint for your Python script to publish a validated bundle
#[hdk_extern]
pub fn create_triangulation_claim(claim: AcousticTriangulationClaim) -> ExternResult<ActionHash> {
    create_entry(EntryTypes::AcousticTriangulationClaim(claim))
}

/// 100% Deterministic Validator.
/// It only looks at the data INSIDE the proposed entry, preventing DHT forks.
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op {
        Op::StoreEntry(store_entry) => {
            // Check if this is our triangulation claim type
            if let Ok(EntryTypes::AcousticTriangulationClaim(claim)) = EntryTypes::try_from(store_entry.entry) {
                
                // 1. Structural Validation: We need at least 3 sensors to triangulate
                if claim.observations.len() < 3 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "Validation failed: At least 3 sensor observations are required to triangulate.".to_string()
                    ));
                }

                // 2. Physical Validation: Check every pair of sensors for speed-of-sound violations
                for i in 0..claim.observations.len() {
                    for j in (i + 1)..claim.observations.len() {
                        let obs_a = &claim.observations[i];
                        let obs_b = &claim.observations[j];

                        // Skip self-comparison (just in case)
                        if obs_a.sensor_id == obs_b.sensor_id {
                            return Ok(ValidateCallbackResult::Invalid(
                                "Validation failed: Duplicate sensor IDs detected in observations.".to_string()
                            ));
                        }

                        // Calculate physical distance: d = sqrt((dx)^2 + (dy)^2)
                        let dx = obs_a.x_coord - obs_b.x_coord;
                        let dy = obs_a.y_coord - obs_b.y_coord;
                        let distance = (dx * dx + dy * dy).sqrt();

                        // Sound travel limits
                        let max_travel_time = distance / SPEED_OF_SOUND;
                        let actual_time_delta = (obs_a.t_arrival - obs_b.t_arrival).abs();

                        // If the recorded sound took longer to travel between the sensors than physically possible,
                        // then these observations cannot belong to the same acoustic event.
                        if actual_time_delta > (max_travel_time + CLOCK_DRIFT_BUFFER) {
                            return Ok(ValidateCallbackResult::Invalid(format!(
                                "Physical validation failed. Speed of sound limits violated between {} and {}.",
                                obs_a.sensor_id, obs_b.sensor_id
                            )));
                        }
                    }
                }
                return Ok(ValidateCallbackResult::Valid);
            }
            Ok(ValidateCallbackResult::Valid)
        }
        _ => Ok(ValidateCallbackResult::Valid),
    }
}