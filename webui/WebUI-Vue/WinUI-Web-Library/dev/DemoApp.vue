<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import {
  WinButton,
  WinCheckBox,
  WinComboBox,
  WinContentDialog,
  WinExpander,
  WinGrid,
  WinInfoBar,
  WinListView,
  WinNavigationView,
  WinNumberBox,
  WinPasswordBox,
  WinProgressBar,
  WinProgressRing,
  WinScrollViewer,
  WinSlider,
  WinStackPanel,
  WinTextBlock,
  WinTextBox,
  WinToggleButton,
  WinToggleSwitch,
} from '../src'

const isDark = ref(false)
const checked = ref(true)
const triState = ref<boolean | null>(null)
const wifiOn = ref(true)
const count = ref(0)

// 新增控件状态
const dialogOpen = ref(false)
const textValue = ref('Hello, WinUI')
const password = ref('')
const numberValue = ref(10)
const comboValue = ref('Option B')
const sliderValue = ref(65)
const listItems = ref([
  { name: 'Alice', role: 'Designer' },
  { name: 'Bob', role: 'Engineer' },
  { name: 'Carol', role: 'Manager' },
])
const navItems = ref([
  { value: 'home', label: 'Home' },
  { value: 'mail', label: 'Mail' },
  { value: 'calendar', label: 'Calendar' },
])

const themeLabel = computed(() => (isDark.value ? 'Switch to Light' : 'Switch to Dark'))

onMounted(() => {
  document.documentElement.classList.add('theme-light')
})

const toggleTheme = () => {
  isDark.value = !isDark.value
  const el = document.documentElement
  el.classList.toggle('theme-dark', isDark.value)
  el.classList.toggle('theme-light', !isDark.value)
}
</script>

<template>
  <WinStackPanel Class="demo-root" Spacing="20">
    <WinTextBlock FontSize="28" FontWeight="600" Text="WinUI Web Library" />

    <WinTextBlock Text="Buttons" FontWeight="600" />
    <WinStackPanel Orientation="Horizontal" Spacing="8">
      <WinButton Style="AccentButtonStyle" :Content="themeLabel" @Click="toggleTheme" />
      <WinButton Content="Primary" />
      <WinButton Style="SubtleButtonStyle" Content="Subtle" />
      <WinButton Content="Increment" :Background="'#0067C0'" @Click="count++" />
      <WinTextBlock :Text="`count = ${count}`" />
    </WinStackPanel>

    <WinTextBlock Text="ToggleButton" FontWeight="600" />
    <WinStackPanel Orientation="Horizontal" Spacing="8">
      <WinToggleButton Content="Toggle me" />
    </WinStackPanel>

    <WinTextBlock Text="CheckBox / ToggleSwitch" FontWeight="600" />
    <WinStackPanel Orientation="Horizontal" Spacing="24">
      <WinCheckBox v-model="checked" Content="Checked" />
      <WinCheckBox v-model="triState" :IsThreeState="true" Content="Tri-state" />
      <WinToggleSwitch v-model="wifiOn" :Header="wifiOn ? 'Wi-Fi On' : 'Wi-Fi Off'" />
    </WinStackPanel>

    <WinTextBlock Text="Input controls" FontWeight="600" />
    <WinStackPanel Spacing="12" Width="480">
      <WinTextBox :Text="textValue" @update:Text="textValue = $event" Header="TextBox" PlaceholderText="Type something" />
      <WinPasswordBox :Password="password" @update:Password="password = $event" Header="PasswordBox" PlaceholderText="Secret" />
      <WinNumberBox :Value="numberValue" @update:Value="numberValue = $event" Header="NumberBox" :Minimum="0" :Maximum="100" Width="140" />
      <WinComboBox :SelectedValue="comboValue" Header="ComboBox" :ItemsSource="['Option A', 'Option B', 'Option C']" Width="200" />
      <WinSlider :Value="sliderValue" @update:Value="sliderValue = $event" Header="Slider" :Minimum="0" :Maximum="100" />
    </WinStackPanel>

    <WinTextBlock Text="Data / status" FontWeight="600" />
    <WinListView :ItemsSource="listItems" Width="480" Height="160" IsItemClickEnabled>
      <template #item="{ item }">
        <WinStackPanel Spacing="4">
          <WinTextBlock :Text="item.name" FontWeight="600" />
          <WinTextBlock :Text="item.role" FontSize="12" />
        </WinStackPanel>
      </template>
    </WinListView>

    <WinStackPanel Orientation="Horizontal" Spacing="32" VerticalAlignment="Center">
      <WinProgressBar :Value="sliderValue" Width="240" />
      <WinProgressRing IsActive IsIndeterminate Width="32" Height="32" />
    </WinStackPanel>

    <WinInfoBar IsOpen Severity="Success" Title="Saved" Message="Your changes were saved successfully." IsClosable />

    <WinExpander Header="Expander" IsExpanded>
      <WinTextBlock Text="Expanded content inside a WinExpander." />
    </WinExpander>

    <WinTextBlock Text="NavigationView" FontWeight="600" />
    <div class="demo-nav-host">
      <WinNavigationView :MenuItems="navItems" IsPaneToggleButtonVisible IsBackButtonVisible="Collapsed">
        <WinStackPanel Spacing="8">
          <WinTextBlock Text="Selected page" FontSize="20" FontWeight="600" />
          <WinTextBlock Text="This is the NavigationView content pane." />
          <WinButton Content="Open dialog" @Click="dialogOpen = true" />
        </WinStackPanel>
        <template #PaneFooter>
          <WinTextBlock Text="Pane footer" />
        </template>
      </WinNavigationView>
    </div>

    <WinTextBlock Text="Grid + ScrollViewer" FontWeight="600" />
    <WinScrollViewer Width="780" Height="180" VerticalScrollMode="Auto" VerticalScrollBarVisibility="Auto">
      <WinGrid :ColumnDefinitions="'Auto,*'" :ColumnSpacing="12" :RowSpacing="8">
        <WinTextBlock Text="Row 1" FontWeight="600" />
        <WinTextBlock Text="The first row of a scrollable grid. It should be tall enough to force a vertical scrollbar." />

        <WinTextBlock Text="Row 2" FontWeight="600" />
        <WinTextBlock Text="The second row — more content, WrappedTextExample. Keep going to overflow." />

        <WinTextBlock Text="Row 3" FontWeight="600" />
        <WinTextBlock Text="Third row. Scroll down to reveal more of what this control can do." />

        <WinTextBlock Text="Row 4" FontWeight="600" />
        <WinTextBlock Text="Fourth row, bottom of the content." />

        <WinTextBlock Text="Row 5" FontWeight="600" />
        <WinTextBlock Text="Fifth row, the last one, far enough down to demonstrate scrolling." />
      </WinGrid>
    </WinScrollViewer>

    <WinContentDialog
      v-model:IsOpen="dialogOpen"
      Title="Content dialog"
      Content="This dialog is rendered by WinContentDialog."
      PrimaryButtonText="OK"
      CloseButtonText="Cancel"
    />
  </WinStackPanel>
</template>

<style scoped>
.demo-root {
  width: 100%;
  height: 100%;
  padding: 24px 32px;
  box-sizing: border-box;
  overflow: auto;
}

.demo-nav-host {
  width: 100%;
  height: 320px;
  border: 1px solid var(--card-stroke);
  border-radius: 8px;
  overflow: hidden;
}
</style>