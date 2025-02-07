using Microsoft.ML.Data;

namespace MachineLearning.DataModels
{
    public class ElementData
    {
        [LoadColumn(0)]
        public string? Element { get; set; }
        [LoadColumn(1)]
        public string? Line { get; set; }
        [LoadColumn(2)]
        public string? LinePosition { get; set; }
        [LoadColumn(3)]
        public required string ControlId { get; set; }
        [LoadColumn(4)]
        public required string Name { get; set; }
        [LoadColumn(5)]
        public required string CSSClass { get; set; }
        [LoadColumn(6)]
        public required string Value { get; set; }

        [LoadColumn(7)]
        public required string Role { get; set; }
        [LoadColumn(8)]
        public required string Type { get; set; }
        [LoadColumn(9)]
        public string? TabIndex { get; set; }
        [LoadColumn(10)]
        public string? Placeholder { get; set; }
        [LoadColumn(11)]
        public required string Title { get; set; }
        [LoadColumn(12)]
        public required string Href { get; set; }
    }
}
