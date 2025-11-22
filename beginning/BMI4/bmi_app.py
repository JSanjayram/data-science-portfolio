import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

# BMI calculation function
def calculate_bmi(weight, height):
    return weight / (height ** 2)

# BMI category function
def get_bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight", "🔵"
    elif bmi < 25:
        return "Normal", "🟢"
    elif bmi < 30:
        return "Overweight", "🟡"
    else:
        return "Obese", "🔴"

# BMI chart
def create_bmi_chart(current_bmi):
    categories = ['Underweight', 'Normal', 'Overweight', 'Obese']
    ranges = [18.5, 25, 30, 40]
    colors = ['lightblue', 'lightgreen', 'orange', 'red']
    
    fig, ax = plt.subplots(figsize=(10, 2))
    
    # Create horizontal bar chart
    for i, (cat, range_val, color) in enumerate(zip(categories, ranges, colors)):
        start = 0 if i == 0 else ranges[i-1]
        width = range_val - start
        ax.barh(0, width, left=start, color=color, alpha=0.7, label=cat)
    
    # Mark current BMI
    ax.axvline(x=current_bmi, color='black', linestyle='--', linewidth=2, label=f'Your BMI: {current_bmi:.1f}')
    
    ax.set_xlim(15, 40)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('BMI Value')
    ax.set_title('BMI Categories')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_yticks([])
    
    return fig

# Main app
def main():
    st.set_page_config(page_title="BMI Calculator", page_icon="⚖️", layout="wide")
    
    st.title("⚖️ BMI Calculator")
    st.markdown("Calculate your Body Mass Index and track your health status")
    
    # Expert UI/UX Sidebar Design
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 20px 0; background-color: #f0f0f0; border-radius: 10px;'>
            <h2 style='color: #1f77b4; margin: 0;'>⚖️ BMI Calculator</h2>
            <p style='color: #666; font-size: 14px; margin: 5px 0;'>Enter your measurements below</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Unit System Selection with improved UX
        st.markdown("**🌍 Unit System**")
        unit_system = st.radio(
            "Choose your preferred unit system",
            ["🇪🇺 Metric (kg/m)", "🇺🇸 Imperial (lbs/ft)"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Dynamic input based on unit system
        if "Metric" in unit_system:
            st.markdown("**📏 Your Measurements**")
            
            # Weight input with proper layout
            weight = st.number_input(
                "⚖️ Weight (kg)",
                min_value=1.0,
                max_value=300.0,
                value=70.0,
                step=0.1,
                format="%.1f",
                help="Enter your weight in kilograms"
            )
            
            # Height input with proper layout
            height = st.number_input(
                "📏 Height (m)",
                min_value=0.5,
                max_value=3.0,
                value=1.75,
                step=0.01,
                format="%.2f",
                help="Enter your height in meters"
            )
                
        else:
            st.markdown("**📏 Your Measurements**")
            
            # Weight input with proper layout
            weight_lbs = st.number_input(
                "⚖️ Weight (lbs)",
                min_value=1.0,
                max_value=660.0,
                value=154.0,
                step=0.1,
                format="%.1f",
                help="Enter your weight in pounds"
            )
            
            # Height inputs with proper layout
            st.markdown("**Height**")
            height_col1, height_col2 = st.columns(2)
            
            with height_col1:
                height_ft = st.number_input(
                    "Height in feet",
                    min_value=1.0,
                    max_value=10.0,
                    value=5.0,
                    step=1.0,
                    format="%.0f",
                    help="Enter feet (1-10)"
                )
                st.caption("Feet")
            
            with height_col2:
                height_in = st.number_input(
                    "Height in inches",
                    min_value=0.0,
                    max_value=11.0,
                    value=9.0,
                    step=1.0,
                    format="%.0f",
                    help="Enter inches (0-11)"
                )
                st.caption("Inches")
            
            # Convert to metric
            weight = weight_lbs * 0.453592
            height = (height_ft * 12 + height_in) * 0.0254
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Enhanced Calculate Button
        calculate_clicked = st.button(
            "🧮 Calculate My BMI",
            type="primary",
            use_container_width=True,
            help="Click to calculate your BMI and get health insights"
        )
        
        st.markdown("---")
        
        # Quick BMI Reference
        with st.expander("📊 BMI Reference Guide", expanded=False):
            st.markdown("""
            **BMI Categories:**
            
            🔵 **Underweight** < 18.5
            
            🟢 **Normal** 18.5 - 24.9
            
            🟡 **Overweight** 25.0 - 29.9
            
            🔴 **Obese** ≥ 30.0
            """)
        
        # Health Tips
        with st.expander("💡 Quick Health Tips", expanded=False):
            st.markdown("""
            • **Stay Hydrated** - Drink 8 glasses of water daily
            • **Regular Exercise** - 150 minutes moderate activity per week
            • **Balanced Diet** - Include fruits, vegetables, and whole grains
            • **Quality Sleep** - Aim for 7-9 hours nightly
            """)
    
    # Calculate BMI
    if calculate_clicked:
        bmi = calculate_bmi(weight, height)
        category, emoji = get_bmi_category(bmi)
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Your BMI", f"{bmi:.1f}")
        
        with col2:
            st.metric("Category", f"{emoji} {category}")
        
        with col3:
            if category == "Normal":
                st.success("Healthy weight range!")
            elif category == "Underweight":
                st.warning("Consider gaining weight")
            else:
                st.error("Consider weight management")
        
        # BMI Chart
        st.subheader("📈 BMI Chart")
        fig = create_bmi_chart(bmi)
        st.pyplot(fig)
        
        # Health recommendations
        st.subheader("💡 Health Recommendations")
        
        if category == "Underweight":
            st.info("• Increase caloric intake with nutritious foods\n• Consider strength training\n• Consult a healthcare provider")
        elif category == "Normal":
            st.success("• Maintain current lifestyle\n• Regular exercise and balanced diet\n• Keep monitoring your weight")
        elif category == "Overweight":
            st.warning("• Reduce caloric intake moderately\n• Increase physical activity\n• Focus on whole foods")
        else:
            st.error("• Consult healthcare provider immediately\n• Significant lifestyle changes needed\n• Consider professional guidance")
        
        # BMI History (session state)
        if 'bmi_history' not in st.session_state:
            st.session_state.bmi_history = []
        
        st.session_state.bmi_history.append({
            'Date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
            'BMI': bmi,
            'Category': category,
            'Weight': weight,
            'Height': height
        })
        
        # Display history
        if len(st.session_state.bmi_history) > 1:
            st.subheader("📊 BMI History")
            df = pd.DataFrame(st.session_state.bmi_history)
            st.dataframe(df, use_container_width=True)
            
            # BMI trend chart
            if len(df) > 1:
                st.line_chart(df.set_index('Date')['BMI'])
    
    # Information section
    with st.expander("ℹ️ About BMI"):
        st.markdown("""
        **Body Mass Index (BMI)** is a measure of body fat based on height and weight.
        
        **BMI Categories:**
        - **Underweight**: BMI < 18.5
        - **Normal weight**: BMI 18.5-24.9
        - **Overweight**: BMI 25-29.9
        - **Obese**: BMI ≥ 30
        
        **Formula**: BMI = weight(kg) / height(m)²
        
        **Note**: BMI is a screening tool and doesn't diagnose body fatness or health.
        """)

if __name__ == "__main__":
    main()