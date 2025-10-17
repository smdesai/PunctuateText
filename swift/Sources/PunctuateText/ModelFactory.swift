import CoreML
import Foundation

/// Factory functions to create model instances
@available(macOS 15.0, iOS 17.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
internal func loadPunctuateModel(configuration: MLModelConfiguration = MLModelConfiguration())
    throws -> MLModel
{
    if let modelURL = Bundle.module.url(forResource: "Punctuate", withExtension: "mlmodelc") {
        return try MLModel(contentsOf: modelURL, configuration: configuration)
    }

    if let modelURL = Bundle.module.url(
        forResource: "Punctuate", withExtension: "mlmodelc", subdirectory: "Resources")
    {
        return try MLModel(contentsOf: modelURL, configuration: configuration)
    }

    throw NSError(
        domain: "PunctuateText", code: 7,
        userInfo: [NSLocalizedDescriptionKey: "Punctuate model not found in bundle"])
}
